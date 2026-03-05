#!/usr/bin/env julia

using DataFrames
using Dates
using LinearAlgebra
using Logging
using Parquet2
using Random
using Statistics

const HARD_COLUMN_COUNT = 5
const DISTANCE_COLUMNS = [
    "elevation", "slope", "access",
    "fcc0_u", "fcc0_d",
    "fcc5_u", "fcc5_d",
    "fcc10_u", "fcc10_d",
]
const DISTANCE_THRESHOLDS = Float32[200.0, 2.5, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

global MEMORY_LOG_FILE = nothing

luc_matching_columns(start_year::Int) = (
    "luc_$(start_year)",
    "luc_$(start_year - 5)",
    "luc_$(start_year - 10)",
)

function setup_memory_logging(output_folder::String)
    global MEMORY_LOG_FILE = joinpath(output_folder, "memory_usage_log.txt")
    open(MEMORY_LOG_FILE, "w") do io
        println(io, "Memory Usage Log - Started at $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, repeat("=", 50))
        println(io)
    end
end

function _log(msg::AbstractString; to_file::Bool=true)
    @info msg
    if to_file && MEMORY_LOG_FILE !== nothing
        open(MEMORY_LOG_FILE, "a") do io
            println(io, "[$(Dates.format(now(), "HH:MM:SS"))] $(msg)")
        end
    end
end

function calculate_processors_for_k_grid_size(k_grid_size::Int) :: Int
    max_concurrent_pixels = 320_000
    processors = max_concurrent_pixels ÷ max(k_grid_size, 1)
    return max(1, min(32, processors))
end

function read_parquet_df(path::String)
    return DataFrame(Parquet2.Dataset(path; parallel_column_loading=true, parallel_page_loading=true))
end

function write_parquet_df(path::String, df::DataFrame)
    Parquet2.writefile(path, df)
end

function extract_k_grid_number(filepath::String)
    basename = splitpath(filepath)[end]
    m = match(r"k_(\d+)\.parquet$", basename)
    return m === nothing ? typemax(Int) : parse(Int, m.captures[1])
end

function get_k_grid_id_from_path(k_filepath::String)
    fname = splitpath(k_filepath)[end]
    m = match(r"\d+", fname)
    return m === nothing ? replace(fname, ".parquet" => "") : String(m.match)
end

function empty_like(df::DataFrame)
    cols = [Vector{eltype(df[!, c])}() for c in names(df)]
    return DataFrame(cols, names(df))
end

function to_float32_matrix(df::DataFrame, cols::Vector{String})
    return Matrix{Float32}(df[:, Symbol.(cols)])
end

function to_int32_matrix(df::DataFrame, cols::Vector{String})
    return Matrix{Int32}(df[:, Symbol.(cols)])
end

function make_s_set_mask(
    m_dist_thresholded::Matrix{Float32},
    k_subset_dist_thresholded::Matrix{Float32},
    m_dist_hard::Matrix{Int32},
    k_subset_dist_hard::Matrix{Int32},
    starting_positions::Vector{Int},
    max_potential_matches::Int,
)
    m_size = size(m_dist_thresholded, 1)
    k_size = size(k_subset_dist_thresholded, 1)

    s_include = falses(m_size)
    k_miss = falses(k_size)

    for k in 1:k_size
        matches = 0
        @views k_row = k_subset_dist_thresholded[k, :]
        @views k_hard = k_subset_dist_hard[k, :]

        for idx in 1:m_size
            m_index = mod1(idx + starting_positions[k], m_size)
            @views m_row = m_dist_thresholded[m_index, :]
            @views m_hard = m_dist_hard[m_index, :]

            hard_equals = true
            @inbounds for j in 1:length(m_hard)
                if m_hard[j] != k_hard[j]
                    hard_equals = false
                    break
                end
            end

            if !hard_equals
                continue
            end

            within = true
            @inbounds for j in 1:length(m_row)
                if abs(m_row[j] - k_row[j]) > 1f0
                    within = false
                    break
                end
            end

            if within
                s_include[m_index] = true
                matches += 1
            end

            if matches == max_potential_matches
                break
            end
        end

        k_miss[k] = matches == 0
    end

    return s_include, k_miss
end

function hard_key_from_match_row(row::AbstractVector{Float32})
    return (
        Int32(row[1]), Int32(row[2]), Int32(row[3]), Int32(row[4]), Int32(row[5])
    )
end

function build_categorical_index(s_set_match::Matrix{Float32}, k_subset_match::Matrix{Float32})
    dict = Dict{NTuple{5, Int32}, Vector{Int}}()

    for s_idx in 1:size(s_set_match, 1)
        key = hard_key_from_match_row(@view s_set_match[s_idx, :])
        push!(get!(dict, key, Int[]), s_idx)
    end

    k_to_s_indices = Vector{Vector{Int}}(undef, size(k_subset_match, 1))
    for k_idx in 1:size(k_subset_match, 1)
        key = hard_key_from_match_row(@view k_subset_match[k_idx, :])
        k_to_s_indices[k_idx] = get(dict, key, Int[])
    end

    return k_to_s_indices
end

function mahalanobis_sq(candidate::AbstractVector{Float32}, target::AbstractVector{Float32}, invcov::Matrix{Float32})
    diff = Float32.(candidate .- target)
    return dot(diff, invcov * diff)
end

function greedy_match_with_shuffled_k(
    k_subset_match::Matrix{Float32},
    s_set_match::Matrix{Float32},
    invcov::Matrix{Float32},
    rng::AbstractRNG,
)
    k_order = collect(1:size(k_subset_match, 1))
    shuffle!(rng, k_order)

    k_to_s_indices = build_categorical_index(s_set_match, k_subset_match)
    s_available = trues(size(s_set_match, 1))

    results = Tuple{Int, Int}[]
    matchless = Int[]

    for k_idx in k_order
        candidates = k_to_s_indices[k_idx]
        if isempty(candidates)
            push!(matchless, k_idx)
            continue
        end

        best_s = 0
        best_d = Inf32
        @views k_cont = k_subset_match[k_idx, HARD_COLUMN_COUNT+1:end]

        for s_idx in candidates
            if !s_available[s_idx]
                continue
            end
            @views s_cont = s_set_match[s_idx, HARD_COLUMN_COUNT+1:end]
            d = mahalanobis_sq(s_cont, k_cont, invcov)
            if d < best_d
                best_d = d
                best_s = s_idx
            end
        end

        if best_s == 0
            push!(matchless, k_idx)
        else
            push!(results, (k_idx, best_s))
            s_available[best_s] = false
        end
    end

    return results, matchless
end

function calculate_smd(k_values::Vector{Float64}, s_values::Vector{Float64})
    if isempty(k_values) || isempty(s_values)
        return NaN
    end

    mean_k = mean(k_values)
    mean_s = mean(s_values)

    var_k = length(k_values) > 1 ? var(k_values; corrected=true) : 0.0
    var_s = length(s_values) > 1 ? var(s_values; corrected=true) : 0.0

    n_k = length(k_values)
    n_s = length(s_values)
    pooled_var = ((n_k - 1) * var_k + (n_s - 1) * var_s) / max(n_k + n_s - 2, 1)
    pooled_std = sqrt(pooled_var)

    pooled_std == 0.0 && return 0.0
    return (mean_k - mean_s) / pooled_std
end

function smd_interpretation(smd::Float64)
    if isnan(smd)
        return "No data"
    end
    x = abs(smd)
    x < 0.1 && return "Negligible"
    x < 0.2 && return "Small"
    x < 0.5 && return "Medium"
    x < 0.8 && return "Large"
    return "Very large"
end

function write_smd_csv(path::String, df::DataFrame)
    open(path, "w") do io
        println(io, "variable,n_k,n_s,mean_k,mean_s,std_k,std_s,smd,abs_smd,smd_interpretation")
        for row in eachrow(df)
            println(io,
                string(
                    row.variable, ",",
                    row.n_k, ",",
                    row.n_s, ",",
                    row.mean_k, ",",
                    row.mean_s, ",",
                    row.std_k, ",",
                    row.std_s, ",",
                    row.smd, ",",
                    row.abs_smd, ",",
                    row.smd_interpretation,
                )
            )
        end
    end
end

function analyse_matching_balance(output_folder::String)
    all_files = readdir(output_folder; join=true)
    match_files = filter(f -> endswith(f, ".parquet") && !endswith(f, "_matchless.parquet"), all_files)

    isempty(match_files) && return DataFrame()

    all_k = Dict(var => Float64[] for var in DISTANCE_COLUMNS)
    all_s = Dict(var => Float64[] for var in DISTANCE_COLUMNS)

    for f in match_files
        df = read_parquet_df(f)
        nrow(df) == 0 && continue

        for var in DISTANCE_COLUMNS
            k_col = Symbol("k_" * var)
            s_col = Symbol("s_" * var)
            if k_col in names(df) && s_col in names(df)
                append!(all_k[var], Float64.(skipmissing(df[!, k_col])))
                append!(all_s[var], Float64.(skipmissing(df[!, s_col])))
            end
        end
    end

    rows = NamedTuple[]
    for var in DISTANCE_COLUMNS
        k_vals = all_k[var]
        s_vals = all_s[var]
        if !isempty(k_vals) && !isempty(s_vals)
            smd = calculate_smd(k_vals, s_vals)
            std_k = length(k_vals) > 1 ? std(k_vals; corrected=true) : NaN
            std_s = length(s_vals) > 1 ? std(s_vals; corrected=true) : NaN
            push!(rows, (
                variable=var,
                n_k=length(k_vals),
                n_s=length(s_vals),
                mean_k=mean(k_vals),
                mean_s=mean(s_vals),
                std_k=std_k,
                std_s=std_s,
                smd=smd,
                abs_smd=abs(smd),
                smd_interpretation=smd_interpretation(smd),
            ))
        else
            push!(rows, (
                variable=var,
                n_k=length(k_vals),
                n_s=length(s_vals),
                mean_k=NaN,
                mean_s=NaN,
                std_k=NaN,
                std_s=NaN,
                smd=NaN,
                abs_smd=NaN,
                smd_interpretation="No data",
            ))
        end
    end

    return DataFrame(rows)
end

function create_m_samples(
    m_parquet_filename::String,
    output_folder::String,
    num_samples::Int,
    sample_size::Int,
    seeds::Vector{Int},
)
    m_samples_dir = joinpath(output_folder, "m_samples")
    mkpath(m_samples_dir)

    missing_samples = Int[]
    for i in 1:num_samples
        f = joinpath(m_samples_dir, "m_sample_$(lpad(i, 3, '0')).parquet")
        if !isfile(f)
            push!(missing_samples, i)
        end
    end

    isempty(missing_samples) && return

    _log("Loading full M set once for sample creation")
    m_set = read_parquet_df(m_parquet_filename)
    n = nrow(m_set)

    for i in missing_samples
        rng = MersenneTwister(seeds[i])
        replace = n < sample_size
        idxs = replace ? rand(rng, 1:n, sample_size) : randperm(rng, n)[1:sample_size]

        sample = m_set[idxs, :]
        perm = randperm(rng, length(idxs))
        sample = sample[perm, :]

        sample_filename = joinpath(m_samples_dir, "m_sample_$(lpad(i, 3, '0')).parquet")
        write_parquet_df(sample_filename, sample)
    end
end

function find_match_iteration(
    m_sample_filename::String,
    start_year::Int,
    evaluation_year::Int,
    output_folder::String,
    k_grid_filepath::String,
    seed::Int,
    shuffle_seed::Int,
    max_potential_matches::Int,
)
    _ = evaluation_year

    k_grid_id = get_k_grid_id_from_path(k_grid_filepath)
    _log("Starting K grid $(k_grid_id)")

    rng = MersenneTwister(seed)

    k_subset = read_parquet_df(k_grid_filepath)
    m_set = read_parquet_df(m_sample_filename)

    shuffle_rng = MersenneTwister(shuffle_seed)
    shuf = randperm(shuffle_rng, nrow(m_set))
    m_set = m_set[shuf, :]

    m_dist_thresholded = to_float32_matrix(m_set, DISTANCE_COLUMNS) ./ permutedims(DISTANCE_THRESHOLDS)
    k_subset_dist_thresholded = to_float32_matrix(k_subset, DISTANCE_COLUMNS) ./ permutedims(DISTANCE_THRESHOLDS)

    luc0, luc5, luc10 = luc_matching_columns(start_year)
    luc_columns = filter(c -> startswith(c, "luc"), String.(names(m_set)))
    hard_match_columns = ["country", "ecoregion", luc10, luc5, luc0]

    m_dist_hard = to_int32_matrix(m_set, hard_match_columns)
    k_subset_dist_hard = to_int32_matrix(k_subset, hard_match_columns)

    starting_positions = rand(rng, 0:(size(m_dist_thresholded, 1)-1), size(k_subset_dist_thresholded, 1))
    s_set_mask_true, no_potentials = make_s_set_mask(
        m_dist_thresholded,
        k_subset_dist_thresholded,
        m_dist_hard,
        k_subset_dist_hard,
        starting_positions,
        max_potential_matches,
    )

    s_set = m_set[s_set_mask_true, :]
    potentials = .!no_potentials
    k_subset = k_subset[potentials, :]

    results_df = DataFrame()
    if nrow(s_set) > 0 && nrow(k_subset) > 0
        s_cov_mat = Matrix{Float64}(s_set[:, Symbol.(DISTANCE_COLUMNS)])
        covar = cov(s_cov_mat; dims=1)
        nvars = size(covar, 1)
        reg = Matrix{Float64}(I, nvars, nvars)
        invcov = Matrix{Float32}(inv(covar + 1e-6 * reg))

        match_cols = vcat(hard_match_columns, DISTANCE_COLUMNS)
        s_set_match = Matrix{Float32}(s_set[:, Symbol.(match_cols)])
        k_subset_match = Matrix{Float32}(k_subset[:, Symbol.(match_cols)])

        add_results, _k_idx_matchless = greedy_match_with_shuffled_k(k_subset_match, s_set_match, invcov, rng)

        rows = NamedTuple[]
        for (k_idx, s_idx) in add_results
            k_row = k_subset[k_idx, :]
            s_row = s_set[s_idx, :]

            record = Dict{Symbol, Any}()
            record[:k_lat] = k_row.lat
            record[:k_lng] = k_row.lng
            for c in vcat(luc_columns, DISTANCE_COLUMNS)
                record[Symbol("k_" * c)] = k_row[Symbol(c)]
            end
            record[:s_lat] = s_row.lat
            record[:s_lng] = s_row.lng
            for c in vcat(luc_columns, DISTANCE_COLUMNS)
                record[Symbol("s_" * c)] = s_row[Symbol(c)]
            end
            push!(rows, (; record...))
        end

        results_df = isempty(rows) ? DataFrame() : DataFrame(rows)
    end

    if nrow(results_df) > 0
        write_parquet_df(joinpath(output_folder, "$(k_grid_id).parquet"), results_df)
    end

    matchless_df = empty_like(k_subset)
    write_parquet_df(joinpath(output_folder, "$(k_grid_id)_matchless.parquet"), matchless_df)

    return k_grid_id
end

function find_pairs(
    k_directory::String,
    m_parquet_filename::String,
    start_year::Int,
    evaluation_year::Int,
    seed::Int,
    output_folder::String,
    batch_size::Int,
    processes_count::Int,
    max_potential_matches::Int,
)
    _ = processes_count

    mkpath(output_folder)
    setup_memory_logging(output_folder)

    if !isfile(m_parquet_filename)
        error("M set file not found: $(m_parquet_filename)")
    end

    k_grid_files_unsorted = filter(f -> occursin(r"k_.*\.parquet$", splitpath(f)[end]), readdir(k_directory; join=true))
    isempty(k_grid_files_unsorted) && error("No k_*.parquet files found in directory: $(k_directory)")

    k_grid_files = sort(k_grid_files_unsorted; by=extract_k_grid_number)
    num_k_grids = length(k_grid_files)

    first_k_grid = read_parquet_df(k_grid_files[1])
    first_k_grid_size = nrow(first_k_grid)
    optimal_processors = calculate_processors_for_k_grid_size(first_k_grid_size)
    _log("Calculated optimal processor count: $(optimal_processors)")

    rng = MersenneTwister(seed)
    iteration_seeds = rand(rng, 0:1_999_999, num_k_grids)
    shuffle_seeds = rand(rng, 0:1_999_999, num_k_grids)

    m_set = read_parquet_df(m_parquet_filename)
    total_m_rows = nrow(m_set)
    sample_size = max(5_000_000, Int(floor(0.1 * total_m_rows)))

    create_m_samples(
        m_parquet_filename,
        output_folder,
        num_k_grids,
        sample_size,
        Int.(iteration_seeds),
    )

    m_samples_dir = joinpath(output_folder, "m_samples")
    processed = String[]

    for i in 1:batch_size:num_k_grids
        batch_end = min(i + batch_size - 1, num_k_grids)
        for j in i:batch_end
            m_sample_file = joinpath(m_samples_dir, "m_sample_$(lpad(j, 3, '0')).parquet")
            kid = find_match_iteration(
                m_sample_file,
                start_year,
                evaluation_year,
                output_folder,
                k_grid_files[j],
                Int(iteration_seeds[j]),
                Int(shuffle_seeds[j]),
                max_potential_matches,
            )
            push!(processed, kid)
        end
        _log("Processed $(length(processed))/$(num_k_grids) K grids")
    end

    smd_df = analyse_matching_balance(output_folder)
    if nrow(smd_df) > 0
        write_smd_csv(joinpath(output_folder, "matching_balance_smd.csv"), smd_df)
    end
end

function parse_cli_args(args::Vector{String})
    params = Dict{String, String}(
        "batch_size" => "16",
        "processes_count" => "16",
        "seed" => "42",
        "max_potential_matches" => "1000",
    )

    i = 1
    while i <= length(args)
        token = args[i]
        if startswith(token, "--")
            key = replace(token, "--" => "")
            if i == length(args) || startswith(args[i+1], "--")
                error("Missing value for argument --$(key)")
            end
            params[key] = args[i+1]
            i += 2
        else
            error("Unexpected argument: $(token)")
        end
    end

    required = ["k_directory", "m_parquet_filename", "start_year", "evaluation_year", "output_folder"]
    for r in required
        haskey(params, r) || error("Missing required argument --$(r)")
    end

    return (
        k_directory = params["k_directory"],
        m_parquet_filename = params["m_parquet_filename"],
        start_year = parse(Int, params["start_year"]),
        evaluation_year = parse(Int, params["evaluation_year"]),
        output_folder = params["output_folder"],
        batch_size = parse(Int, params["batch_size"]),
        processes_count = parse(Int, params["processes_count"]),
        seed = parse(Int, params["seed"]),
        max_potential_matches = parse(Int, params["max_potential_matches"]),
    )
end

function main()
    Logging.global_logger(ConsoleLogger(stderr, Logging.Info))
    args = parse_cli_args(ARGS)

    find_pairs(
        args.k_directory,
        args.m_parquet_filename,
        args.start_year,
        args.evaluation_year,
        args.seed,
        args.output_folder,
        args.batch_size,
        args.processes_count,
        args.max_potential_matches,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
