#!/usr/bin/env julia

using DataFrames
using Dates
using Distributed
using GLM
using LinearAlgebra
using Logging
using Parquet2
using Random
using Statistics
using Arrow  # Much faster than Parquet2 for intermediate files
using CodecZstd

const DISTANCE_COLUMNS = [
    "elevation", "slope", "access",
    "fcc0_u", "fcc0_d",
    "fcc5_u", "fcc5_d",
    "fcc10_u", "fcc10_d",
]

const THRESHOLDS = Float32[200.0, 2.5, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

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

function calculate_processors_for_k_grid_size(k_grid_size::Int)
    num_processors = 32
    println("Calculated $(num_processors) processors for K grid size $(k_grid_size)")
    return num_processors
end

function unique_preserve_order(items::Vector{String})
    seen = Set{String}()
    out = String[]
    for item in items
        if !(item in seen)
            push!(seen, item)
            push!(out, item)
        end
    end
    return out
end

function read_parquet_df(path::String)
    return DataFrame(Parquet2.Dataset(path; parallel_column_loading=true, parallel_page_loading=true))
end

function parquet_columns(path::String)
    try
        return String.(names(Parquet2.Dataset(path)))
    catch
        return String[]
    end
end

function read_parquet_selected(path::String, cols::Vector{String})
    df = read_parquet_df(path)
    if isempty(cols)
        return df
    end
    present = Symbol.(filter(c -> c in String.(names(df)), cols))
    return select(df, present; copycols=false)
end

function write_parquet_df(path::String, df::DataFrame)
    Parquet2.writefile(path, df)
end

function write_arrow_df(path::String, df::DataFrame)
    Arrow.write(path, df, compress=ZstdCompressor())
end

function read_arrow_df(path::String)
    return DataFrame(Arrow.Table(path))
end

function read_table_selected(path::String, cols::Vector{String})
    if endswith(lowercase(path), ".arrow")
        df = read_arrow_df(path)
        if isempty(cols)
            return df
        end
        present = Symbol.(filter(c -> c in String.(names(df)), cols))
        return df[:, present]
    end
    return read_parquet_selected(path, cols)
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

function k_outputs_exist(k_filepath::String, output_folder::String)
    kid = get_k_grid_id_from_path(k_filepath)
    p1 = joinpath(output_folder, "$(kid).parquet")
    p2 = joinpath(output_folder, "$(kid)_matchless.parquet")
    return isfile(p1) && filesize(p1) > 0 && isfile(p2) && filesize(p2) > 0
end

function build_hard_keys(hard::Matrix{Int32})
    p = Int64(100_000)
    keys = zeros(Int64, size(hard, 1))
    for c in 1:size(hard, 2)
        keys .+= Int64.(hard[:, c]) .* (p ^ (c - 1))
    end
    return keys
end

function threshold_match_group(
    k_row_dist::AbstractVector{Float32},
    group_dist::Matrix{Float32},
    group_global_indices::Vector{Int},
    start_offset::Int,
)
    g_size = size(group_dist, 1)
    out = Vector{Int}(undef, g_size)
    matches = 0

    for idx in 1:g_size
        g_idx = mod1(idx + start_offset, g_size)
        ok = true
        @inbounds for j in 1:size(group_dist, 2)
            if abs(group_dist[g_idx, j] - k_row_dist[j]) > 1f0
                ok = false
                break
            end
        end
        if ok
            matches += 1
            out[matches] = group_global_indices[g_idx]
        end
    end

    return out[1:matches], matches
end

function compute_propensity_scores(
    m_set::DataFrame,
    k_subset::DataFrame,
    hard_cols::Vector{String},
)
    features = String[]
    for c in vcat(hard_cols, DISTANCE_COLUMNS)
        if c in String.(names(m_set)) && c in String.(names(k_subset))
            push!(features, c)
        end
    end

    if isempty(features)
        error("No overlapping features found between M and K for propensity computation")
    end

    x_m = Matrix{Float64}(m_set[:, Symbol.(features)])
    x_k = Matrix{Float64}(k_subset[:, Symbol.(features)])
    x = vcat(x_m, x_k)
    y = vcat(zeros(Float64, size(x_m, 1)), ones(Float64, size(x_k, 1)))

    mu = vec(mean(x, dims=1))
    sigma = vec(std(x, dims=1))
    sigma[sigma .== 0.0] .= 1.0
    x_scaled = (x .- permutedims(mu)) ./ permutedims(sigma)

    try
        model = glm(x_scaled, y, Binomial(), LogitLink())
        scores = Float32.(predict(model, x_scaled))
        return scores[1:size(x_m, 1)], scores[size(x_m, 1)+1:end]
    catch err
        @warn "Propensity model fit failed; falling back to constant scores" exception=(err, catch_backtrace())
        m_scores = fill(Float32(0.5), size(x_m, 1))
        k_scores = fill(Float32(0.5), size(x_k, 1))
        return m_scores, k_scores
    end
end

@inline function _sigmoid_clamped(x::Float64)
    z = clamp(x, -30.0, 30.0)
    return 1.0 / (1.0 + exp(-z))
end

function local_propensity_weights(
    k_row::AbstractVector{Float32},
    cand_rows::AbstractMatrix{Float32};
    ridge::Float64=1.0,
    max_iter::Int=6,
)
    n = size(cand_rows, 1)
    p = size(cand_rows, 2)

    n == 0 && return Float32[]
    n == 1 && return Float32[1.0f0]

    x = Matrix{Float64}(undef, n + 1, p + 1)
    x[:, 1] .= 1.0
    @views x[1:n, 2:end] .= Float64.(cand_rows)
    @views x[n+1, 2:end] .= Float64.(k_row)

    y = zeros(Float64, n + 1)
    y[n + 1] = 1.0

    beta = zeros(Float64, p + 1)
    eta = zeros(Float64, n + 1)
    prob = similar(eta)
    w = similar(eta)
    z = similar(eta)

    # Small, regularised IRLS loop (much lighter than calling GLM per K pixel)
    for _ in 1:max_iter
        mul!(eta, x, beta)
        @inbounds for i in 1:(n + 1)
            prob[i] = _sigmoid_clamped(eta[i])
            w[i] = prob[i] * (1.0 - prob[i]) + 1e-6
            z[i] = eta[i] + (y[i] - prob[i]) / w[i]
        end

        xw = x .* w
        a = transpose(x) * xw
        b = transpose(x) * (w .* z)
        @inbounds for j in 2:(p + 1)
            a[j, j] += ridge
        end

        try
            beta .= a \ b
        catch
            # Fallback to equal weighting if solve is unstable
            return fill(Float32(1.0 / n), n)
        end
    end

    mul!(eta, x, beta)
    @inbounds for i in 1:(n + 1)
        prob[i] = _sigmoid_clamped(eta[i])
    end

    k_prob = prob[n + 1]
    cand_prob = @view prob[1:n]
    diffs = abs.(cand_prob .- k_prob) .+ 1e-6
    weights = 1.0 ./ diffs
    wsum = sum(weights)
    if !isfinite(wsum) || wsum <= 0.0
        return fill(Float32(1.0 / n), n)
    end

    weights ./= wsum
    return Float32.(weights)
end

function propensity_gap_weights(k_score::Float32, cand_scores::AbstractVector{Float32})
    n = length(cand_scores)
    n == 0 && return Float32[]
    n == 1 && return Float32[1.0f0]

    weights = 1.0 ./ (abs.(cand_scores .- k_score) .+ 1e-6)
    wsum = sum(weights)
    if !isfinite(wsum) || wsum <= 0.0
        return fill(Float32(1.0 / n), n)
    end
    return Float32.(weights ./ wsum)
end

function mahalanobis_kernel_weights(
    k_row::AbstractVector{Float32},
    cand_rows::AbstractMatrix{Float32},
    invcov::AbstractMatrix{Float32},
    bandwidth::Float64,
)
    n = size(cand_rows, 1)
    n == 0 && return Float32[]
    n == 1 && return Float32[1.0f0]

    bw2 = max(bandwidth, 1e-6)^2
    weights = Vector{Float64}(undef, n)
    tmp = Vector{Float64}(undef, size(cand_rows, 2))
    invcov64 = Matrix{Float64}(invcov)

    @inbounds for i in 1:n
        for j in 1:length(tmp)
            tmp[j] = Float64(cand_rows[i, j] - k_row[j])
        end
        d2 = dot(tmp, invcov64 * tmp)
        weights[i] = exp(-0.5 * d2 / bw2)
    end

    wsum = sum(weights)
    if !isfinite(wsum) || wsum <= 0.0
        return fill(Float32(1.0 / n), n)
    end

    return Float32.(weights ./ wsum)
end

function blend_weights(
    w_a::AbstractVector{Float32},
    w_b::AbstractVector{Float32},
    lambda::Float64,
)
    n = length(w_a)
    n == 0 && return Float32[]
    n == 1 && return Float32[1.0f0]

    λ = clamp(lambda, 0.0, 1.0)
    out = λ .* Float64.(w_a) .+ (1.0 - λ) .* Float64.(w_b)
    s = sum(out)
    if !isfinite(s) || s <= 0.0
        return fill(Float32(1.0 / n), n)
    end
    return Float32.(out ./ s)
end

function make_s_set_mask_grouped(
    m_dist_thresholded::Matrix{Float32},
    k_subset_dist_thresholded::Matrix{Float32},
    m_hard_df::Matrix{Int32},
    k_hard_df::Matrix{Int32},
    m_propensity::Vector{Float32},
    k_propensity::Vector{Float32},
    rng::AbstractRNG,
    min_potential_matches::Int,
    max_potential_matches::Int;
    grid_id::String="",
)
    t_start = time()
    k_size = size(k_subset_dist_thresholded, 1)
    k_to_m_indices = fill(-1, k_size, max_potential_matches)
    k_match_counts = zeros(Int, k_size)
    k_miss = falses(k_size)

    m_keys = build_hard_keys(m_hard_df)
    grouped = Dict{Int64, Vector{Int}}()
    for (idx, key) in enumerate(m_keys)
        push!(get!(grouped, key, Int[]), idx)
    end

    group_sizes = length.(values(grouped))
    med_group = isempty(group_sizes) ? 0 : Int(round(median(group_sizes)))
    max_group = isempty(group_sizes) ? 0 : maximum(group_sizes)
    println("[prop] K grid $(grid_id): grouped M into $(length(grouped)) hard-key groups (median size $(med_group), max $(max_group)) in $(round(time()-t_start, digits=2))s")

    group_cache = Dict{Int64, Tuple{Matrix{Float32}, Vector{Int}}}()
    for (key, idxs) in grouped
        group_cache[key] = (m_dist_thresholded[idxs, :], idxs)
    end

    k_keys = build_hard_keys(k_hard_df)
    n_no_group = 0
    t_loop = time()

    for k in 1:k_size
        key = k_keys[k]
        if !haskey(group_cache, key)
            k_miss[k] = true
            n_no_group += 1
            continue
        end

        group_dist, group_global_idx = group_cache[key]
        g_size = size(group_dist, 1)
        start_offset = g_size > 0 ? rand(rng, 0:g_size-1) : 0
        matched_idx, n = threshold_match_group(view(k_subset_dist_thresholded, k, :), group_dist, group_global_idx, start_offset)

        if n == 0
            k_miss[k] = true
        else
            if n > max_potential_matches
                k_score = k_propensity[k]
                score_diffs = abs.(m_propensity[matched_idx] .- k_score)
                keep_order = sortperm(score_diffs)[1:max_potential_matches]
                selected = matched_idx[keep_order]
            else
                selected = matched_idx
            end

            n_sel = length(selected)
            if n_sel < min_potential_matches
                k_miss[k] = true
            else
                k_to_m_indices[k, 1:n_sel] .= selected
                k_match_counts[k] = n_sel
            end
        end

        if k % 200 == 0
            println("[prop] K grid $(grid_id): matched $(k)/$(k_size) K pixels ($(round(time()-t_loop, digits=1))s elapsed)")
        end
    end

    println("[prop] K grid $(grid_id): matching loop done in $(round(time()-t_loop, digits=2))s ($(n_no_group) K pixels had no group in M)")
    return k_miss, k_to_m_indices, k_match_counts
end

function find_proportion_iteration(
    m_sample_filename::String,
    start_year::Int,
    evaluation_year::Int,
    output_folder::String,
    k_grid_filepath::String,
    seed::Int,
    shuffle_seed::Int,
    min_potential_matches::Int,
    max_potential_matches::Int,
    similarity_mode::String,
    propensity_caliper::Float64,
    mahalanobis_lambda::Float64,
    mahalanobis_bandwidth::Float64,
    cluster_k_first::Bool,
)
    k_grid_filename = splitpath(k_grid_filepath)[end]
    m = match(r"\d+", k_grid_filename)
    k_grid_id = m === nothing ? replace(k_grid_filename, ".parquet" => "") : String(m.match)

    println("[prop] Starting K grid $(k_grid_id)")

    rng = MersenneTwister(seed)
    luc_years = collect(start_year:evaluation_year)
    luc_col_names = ["luc_$(y)" for y in luc_years]
    luc0, luc5, luc10 = luc_matching_columns(start_year)
    k_hard_cols = ["country", "ecoregion", luc10, luc5, luc0]
    start_luc_col = "luc_$(start_year)"

    needed_k_cols = unique_preserve_order(vcat(DISTANCE_COLUMNS, ["lat", "lng"], k_hard_cols, luc_col_names))
    k_all = read_parquet_selected(k_grid_filepath, needed_k_cols)

    if !(start_luc_col in String.(names(k_all)))
        error("K grid $(k_grid_id): required column '$(start_luc_col)' not found")
    end

    k_subset = k_all[k_all[:, Symbol(start_luc_col)] .== 1, :]
    n_dropped = nrow(k_all) - nrow(k_subset)

    if nrow(k_subset) == 0
        println("[prop] K grid $(k_grid_id): no undisturbed pixels – skipping")
        write_parquet_df(joinpath(output_folder, "$(k_grid_id).parquet"), DataFrame())
        write_parquet_df(joinpath(output_folder, "$(k_grid_id)_matchless.parquet"), DataFrame())
        return k_grid_id
    end

    println("[prop] K grid $(k_grid_id): $(nrow(k_subset)) undisturbed pixels ($(n_dropped) dropped)")
    println("[prop] K grid $(k_grid_id): reading M sample from $(splitpath(m_sample_filename)[end])...")

    m_available_cols = parquet_columns(m_sample_filename)
    prop_cols = [c for c in m_available_cols if occursin("prop", lowercase(c))]
    needed_m_cols = unique_preserve_order(vcat(DISTANCE_COLUMNS, ["lat", "lng"], k_hard_cols, luc_col_names, prop_cols))
    m_set = read_table_selected(m_sample_filename, needed_m_cols)

    if start_luc_col in String.(names(m_set))
        m_set = m_set[m_set[:, Symbol(start_luc_col)] .== 1, :]
    end

    shuf_rng = MersenneTwister(shuffle_seed)
    perm = randperm(shuf_rng, nrow(m_set))
    m_set = m_set[perm, :]

    println("[prop] K grid $(k_grid_id): loaded & shuffled M sample ($(nrow(m_set)) rows)")

    present_dist = [c for c in DISTANCE_COLUMNS if c in String.(names(m_set)) && c in String.(names(k_subset))]
    m_dist_raw = Matrix{Float32}(m_set[:, Symbol.(present_dist)])
    k_dist_raw = Matrix{Float32}(k_subset[:, Symbol.(present_dist)])
    scale = THRESHOLDS[1:length(present_dist)]
    m_dist = m_dist_raw ./ permutedims(scale)
    k_dist = k_dist_raw ./ permutedims(scale)

    n_dist = length(present_dist)
    invcov = Matrix{Float32}(I, n_dist, n_dist)
    if n_dist > 0 && size(m_dist, 1) > 1
        covar = cov(Float64.(m_dist); dims=1)
        if all(isfinite, covar)
            reg = Matrix{Float64}(I, n_dist, n_dist)
            invcov = Float32.(inv(covar + 1e-6 * reg))
        end
    end

    for hc in k_hard_cols
        if !(hc in String.(names(m_set))) || !(hc in String.(names(k_subset)))
            error("Missing hard-match column $(hc) in M or K")
        end
    end

    m_propensity, k_propensity = compute_propensity_scores(m_set, k_subset, k_hard_cols)
    m_hard = Matrix{Int32}(m_set[:, Symbol.(k_hard_cols)])
    k_hard = Matrix{Int32}(k_subset[:, Symbol.(k_hard_cols)])

    # Features used for per-pixel local propensity weighting
    local_prop_features = [
        c for c in vcat(k_hard_cols, DISTANCE_COLUMNS)
        if c in String.(names(m_set)) && c in String.(names(k_subset))
    ]
    if isempty(local_prop_features)
        error("No overlapping local propensity features found between M and K")
    end
    m_local_x = Matrix{Float32}(m_set[:, Symbol.(local_prop_features)])
    k_local_x = Matrix{Float32}(k_subset[:, Symbol.(local_prop_features)])
    mu_local = vec(mean(m_local_x, dims=1))
    sd_local = vec(std(m_local_x, dims=1))
    sd_local[sd_local .== 0.0f0] .= 1.0f0
    m_local_x = (m_local_x .- permutedims(mu_local)) ./ permutedims(sd_local)
    k_local_x = (k_local_x .- permutedims(mu_local)) ./ permutedims(sd_local)

    k_miss, k_to_m_indices, k_match_counts = make_s_set_mask_grouped(
        m_dist,
        k_dist,
        m_hard,
        k_hard,
        m_propensity,
        k_propensity,
        rng,
        min_potential_matches,
        max_potential_matches;
        grid_id=k_grid_id,
    )

    n_matched = count(!, k_miss)
    n_matchless = count(identity, k_miss)
    println("[prop] K grid $(k_grid_id): $(n_matched) matched, $(n_matchless) matchless")

    existing_luc = [(y, "luc_$(y)") for y in luc_years if "luc_$(y)" in String.(names(m_set))]
    m_luc_arrays = Dict{Int, Vector{Int16}}()
    for (y, c) in existing_luc
        m_luc_arrays[y] = Int16.(m_set[:, Symbol(c)])
    end

    k_luc_arrays = Dict{Int, Vector{Int16}}()
    for (y, c) in existing_luc
        if c in String.(names(k_subset))
            k_luc_arrays[y] = Int16.(k_subset[:, Symbol(c)])
        end
    end

    k_lat = Float64.(k_subset[:, :lat])
    k_lng = Float64.(k_subset[:, :lng])
    m_has_coords = ("lat" in String.(names(m_set))) && ("lng" in String.(names(m_set)))
    m_lat = m_has_coords ? Float64.(m_set[:, :lat]) : Float64[]
    m_lng = m_has_coords ? Float64.(m_set[:, :lng]) : Float64[]

    present_dist_out = [c for c in DISTANCE_COLUMNS if c in String.(names(m_set))]
    m_cont_arrays = Dict{String, Vector{Float32}}()
    for c in present_dist_out
        m_cont_arrays[c] = Float32.(m_set[:, Symbol(c)])
    end

    n_k = nrow(k_subset)

    res_n_cand = zeros(Int32, n_k)
    res_matched = trues(n_k)
    res_k_luc = Dict{Int, Vector{Int16}}(y => Vector{Int16}(undef, n_k) for y in keys(k_luc_arrays))
    res_s_prop = Dict{Int, Vector{Float32}}(y => fill(Float32(NaN), n_k) for (y, _) in existing_luc)
    res_wm_cont = Dict{String, Vector{Float32}}(c => fill(Float32(NaN), n_k) for c in present_dist_out)
    res_candidate_points_wkt = fill("", n_k)

    mode = lowercase(similarity_mode)
    k_order = collect(1:n_k)
    if cluster_k_first
        k_keys = build_hard_keys(k_hard)
        k_prop_bin = Int32.(round.(k_propensity .* 1000f0))
        sort!(k_order, by=i -> (k_keys[i], k_prop_bin[i]))
    end

    for k_idx in k_order
        for (y, arr) in k_luc_arrays
            res_k_luc[y][k_idx] = arr[k_idx]
        end

        n_cand = k_match_counts[k_idx]
        if n_cand == 0
            res_n_cand[k_idx] = 0
            res_matched[k_idx] = false
            continue
        end

        cand_idx = k_to_m_indices[k_idx, 1:n_cand]
        if propensity_caliper > 0.0
            keep = abs.(m_propensity[cand_idx] .- k_propensity[k_idx]) .<= propensity_caliper
            cand_idx = cand_idx[keep]
            n_cand = length(cand_idx)
        end

        res_n_cand[k_idx] = Int32(n_cand)

        if n_cand < min_potential_matches
            res_matched[k_idx] = false
            continue
        end

        weights = if mode == "local_propensity"
            cand_rows = @view m_local_x[cand_idx, :]
            k_row = @view k_local_x[k_idx, :]
            local_propensity_weights(k_row, cand_rows)
        elseif mode == "propensity_gap"
            propensity_gap_weights(k_propensity[k_idx], m_propensity[cand_idx])
        elseif mode == "mahalanobis"
            if n_dist == 0
                propensity_gap_weights(k_propensity[k_idx], m_propensity[cand_idx])
            else
                cand_rows_dist = @view m_dist[cand_idx, :]
                k_row_dist = @view k_dist[k_idx, :]
                mahalanobis_kernel_weights(k_row_dist, cand_rows_dist, invcov, mahalanobis_bandwidth)
            end
        else
            if n_dist == 0
                propensity_gap_weights(k_propensity[k_idx], m_propensity[cand_idx])
            else
                cand_rows_dist = @view m_dist[cand_idx, :]
                k_row_dist = @view k_dist[k_idx, :]
                w_mahal = mahalanobis_kernel_weights(k_row_dist, cand_rows_dist, invcov, mahalanobis_bandwidth)
                w_prop = propensity_gap_weights(k_propensity[k_idx], m_propensity[cand_idx])
                blend_weights(w_mahal, w_prop, mahalanobis_lambda)
            end
        end

        for (y, m_arr) in m_luc_arrays
            cand_vals = m_arr[cand_idx]
            und = Float32.(cand_vals .== 1)
            res_s_prop[y][k_idx] = sum(weights .* und)
        end

        for c in present_dist_out
            cvals = m_cont_arrays[c][cand_idx]
            res_wm_cont[c][k_idx] = sum(weights .* cvals)
        end

        if m_has_coords
            io = IOBuffer()
            write(io, "MULTIPOINT(")
            for j in 1:length(cand_idx)
                if j > 1
                    write(io, ",")
                end
                write(io, "(")
                write(io, string(m_lng[cand_idx[j]]))
                write(io, " ")
                write(io, string(m_lat[cand_idx[j]]))
                write(io, ")")
            end
            write(io, ")")
            res_candidate_points_wkt[k_idx] = String(take!(io))
        end
    end

    results_df = DataFrame(k_lat=k_lat, k_lng=k_lng)
    for y in sort(collect(keys(res_k_luc)))
        results_df[!, Symbol("k_luc_$(y)")] = res_k_luc[y]
    end
    for y in sort(collect(keys(res_s_prop)))
        vals = res_s_prop[y]
        rounded = Vector{Float32}(undef, length(vals))
        @inbounds for i in eachindex(vals)
            v = vals[i]
            rounded[i] = isnan(v) ? v : Float32(round(v, digits=5))
        end
        results_df[!, Symbol("s_prop_$(y)")] = rounded
    end
    for c in present_dist_out
        results_df[!, Symbol("s_wmean_$(c)")] = res_wm_cont[c]
    end
    results_df[!, :n_candidates] = res_n_cand
    results_df[!, :s_candidate_points_wkt] = res_candidate_points_wkt

    matched_df = results_df[res_matched, :]
    matchless_df = results_df[.!res_matched, :]

    write_parquet_df(joinpath(output_folder, "$(k_grid_id).parquet"), matched_df)
    write_parquet_df(joinpath(output_folder, "$(k_grid_id)_matchless.parquet"), matchless_df)

    println("[prop] K grid $(k_grid_id): saved $(nrow(matched_df)) matched, $(nrow(matchless_df)) matchless")
    return k_grid_id
end

function _iteration_tuple(args)
    (msf, start_year, evaluation_year, out, kf, iseed, sseed, min_pm, max_pm, similarity_mode, propensity_caliper, mahalanobis_lambda, mahalanobis_bandwidth, cluster_k_first) = args
    return find_proportion_iteration(
        msf,
        start_year,
        evaluation_year,
        out,
        kf,
        iseed,
        sseed,
        min_pm,
        max_pm,
        similarity_mode,
        propensity_caliper,
        mahalanobis_lambda,
        mahalanobis_bandwidth,
        cluster_k_first,
    )
end

# ============================================================================
# OPTIMIZED M SAMPLE CREATION - MUCH FASTER VERSION
# ============================================================================

function create_m_samples_fast(
    m_parquet_filename::String,
    output_folder::String,
    num_samples::Int,
    sample_size::Int,
    seeds::Vector{Int},
    start_year::Int,
)
    m_samples_dir = joinpath(output_folder, "m_samples")
    mkpath(m_samples_dir)

    # Check which samples are missing (accept existing parquet or arrow)
    missing = Int[]
    for i in 1:num_samples
        p_parquet = joinpath(m_samples_dir, "m_sample_$(lpad(i, 3, '0')).parquet")
        p_arrow = joinpath(m_samples_dir, "m_sample_$(lpad(i, 3, '0')).arrow")
        if !isfile(p_parquet) && !isfile(p_arrow)
            push!(missing, i)
        end
    end

    if isempty(missing)
        _log("All M samples already exist. Skipping.")
        return
    end

    _log("⚡ FAST MODE: Creating $(length(missing)) M samples of $(sample_size) rows")
    
    # Determine required columns
    luc0, luc5, luc10 = luc_matching_columns(start_year)
    start_luc_col = "luc_$(start_year)"
    all_cols = parquet_columns(m_parquet_filename)
    luc_cols_all = [c for c in all_cols if startswith(c, "luc_")]
    prop_cols_all = [c for c in all_cols if occursin("prop", lowercase(c))]
    needed = unique_preserve_order(vcat(
        DISTANCE_COLUMNS,
        ["lat", "lng"],
        ["country", "ecoregion", luc0, luc5, luc10, start_luc_col],
        luc_cols_all,
        prop_cols_all,
    ))

    # Cache filtered source table to speed repeated reruns after crashes/restarts.
    source_cache = joinpath(m_samples_dir, "m_source_filtered_$(start_year).arrow")
    m_source = DataFrame()
    if isfile(source_cache)
        _log("Loading cached filtered M source: $(source_cache)")
        m_source = read_arrow_df(source_cache)
        cache_cols = Set(String.(names(m_source)))
        missing_from_cache = [c for c in needed if !(c in cache_cols)]
        if !isempty(missing_from_cache)
            _log("Cached M source missing $(length(missing_from_cache)) required columns; rebuilding cache")
            m_source = DataFrame()
        end
    end

    if nrow(m_source) == 0
        _log("Loading source M data from parquet (this may take a moment)...")
        m_source = read_parquet_selected(m_parquet_filename, needed)

        # Filter to undisturbed in start year
        if start_luc_col in String.(names(m_source))
            m_source = m_source[m_source[:, Symbol(start_luc_col)] .== 1, :]
        end

        # Pre-convert to efficient types ONCE
        for col in DISTANCE_COLUMNS
            if col in String.(names(m_source))
                m_source[!, col] = Float32.(m_source[:, col])
            end
        end

        for col in ["country", "ecoregion", luc0, luc5, luc10, start_luc_col]
            if col in String.(names(m_source))
                m_source[!, col] = Int32.(m_source[:, col])
            end
        end

        for col in [c for c in String.(names(m_source)) if startswith(c, "luc_")]
            if col in String.(names(m_source))
                m_source[!, col] = Int16.(m_source[:, col])
            end
        end

        _log("Writing filtered M source cache: $(source_cache)")
        write_arrow_df(source_cache, m_source)
    end

    n_m = nrow(m_source)
    n_m == 0 && error("No rows available in M after filtering $(start_luc_col)==1")
    _log("Source has $(n_m) rows after filtering")
    
    # Convert to Matrix for faster sampling (DataFrame indexing is slow)
    col_names = String.(names(m_source))
    m_data = [Array(m_source[:, col]) for col in col_names]
    
    _log("Generating $(length(missing)) samples in parallel (parquet-only output)...")

    progress = Threads.Atomic{Int}(0)

    # Generate each sample independently (lower peak memory)
    Threads.@threads for s_idx in 1:length(missing)
        i = missing[s_idx]
        rng = MersenneTwister(seeds[i])
        
        # Sample indices
        replace = n_m < sample_size
        idx = if replace
            rand(rng, 1:n_m, sample_size)
        else
            randperm(rng, n_m)[1:sample_size]
        end
        
        # Build sampled columns
        sampled_cols = Vector{Any}(undef, length(col_names))
        for col_idx in 1:length(col_names)
            col_data = m_data[col_idx]
            sample_col = Vector{eltype(col_data)}(undef, sample_size)
            @inbounds for j in 1:sample_size
                sample_col[j] = col_data[idx[j]]
            end
            sampled_cols[col_idx] = sample_col
        end

        # idx is already random-order for both replacement and no-replacement cases;
        # no second shuffle needed.

        # Build DataFrame and write immediately
        sample_df = DataFrame()
        for col_idx in 1:length(col_names)
            sample_df[!, Symbol(col_names[col_idx])] = sampled_cols[col_idx]
        end

        parquet_path = joinpath(m_samples_dir, "m_sample_$(lpad(i, 3, '0')).parquet")
        write_parquet_df(parquet_path, sample_df)

        done_now = Threads.atomic_add!(progress, 1) + 1
        if done_now % 4 == 0 || done_now == length(missing)
            _log("Completed $(done_now)/$(length(missing)) samples")
        end
    end
    
    _log("✅ All samples created successfully")
end

# Replace the original create_m_samples with the fast version
const create_m_samples = create_m_samples_fast

function parse_cli_args(args::Vector{String})
    opts = Dict{String, String}()
    bool_opts = Set(["--use-fork"])
    i = 1
    while i <= length(args)
        key = args[i]
        if key in bool_opts
            opts[key] = "true"
            i += 1
            continue
        end
        if i == length(args)
            error("Missing value for argument $(key)")
        end
        opts[key] = args[i + 1]
        i += 2
    end
    return opts
end

function require_arg(opts::Dict{String, String}, key::String)
    haskey(opts, key) || error("Missing required argument: $(key)")
    return opts[key]
end

function parse_int_opt(opts::Dict{String, String}, key::String, default::Int)
    haskey(opts, key) ? parse(Int, opts[key]) : default
end

function parse_str_opt(opts::Dict{String, String}, key::String, default::Union{String, Nothing})
    haskey(opts, key) ? opts[key] : default
end

function parse_float_opt(opts::Dict{String, String}, key::String, default::Float64)
    haskey(opts, key) ? parse(Float64, opts[key]) : default
end

function find_pairs(
    k_directory::String,
    m_parquet_filename::String,
    m_sample_folder::Union{String, Nothing},
    start_year::Int,
    evaluation_year::Int,
    seed::Int,
    output_folder::String,
    batch_size::Int,
    processes_count::Int,
    min_potential_matches::Int,
    max_potential_matches::Int,
    m_sample_count::Union{Int, Nothing},
    m_sample_size::Int,
    similarity_mode::String,
    propensity_caliper::Float64,
    mahalanobis_lambda::Float64,
    mahalanobis_bandwidth::Float64,
    cluster_k_first::Bool,
)
    _log("Starting find_pairs_prop (Julia)")
    mkpath(output_folder)
    setup_memory_logging(output_folder)

    isfile(m_parquet_filename) || error("M set not found: $(m_parquet_filename)")
    min_potential_matches > 0 || error("--min_potential_matches must be >= 1")
    max_potential_matches >= min_potential_matches || error("--max_potential_matches must be >= --min_potential_matches")
    propensity_caliper >= 0.0 || error("--propensity_caliper must be >= 0")
    mahalanobis_bandwidth > 0.0 || error("--mahalanobis_bandwidth must be > 0")

    k_files = filter(
        f -> occursin(r"^k_\d+\.parquet$", splitpath(f)[end]),
        readdir(k_directory; join=true),
    )
    println("Found $(length(k_files)) K grid files in $(k_directory)")
    sort!(k_files, by=extract_k_grid_number)
    isempty(k_files) && error("No k_*.parquet in $(k_directory)")

    num_k = length(k_files)
    _log("Found $(num_k) K grid files")

    first_size = nrow(read_parquet_df(k_files[1]))
    processes_count = calculate_processors_for_k_grid_size(first_size)
    _log("Target worker count: $(processes_count)")

    rng = MersenneTwister(seed)
    iter_seeds = rand(rng, 0:1_999_999, num_k)
    shuf_seeds = rand(rng, 0:1_999_999, num_k)

    m_sample_files = String[]
    if m_sample_folder !== nothing && isdir(m_sample_folder)
        _log("Using existing M samples from $(m_sample_folder)")
        m_sample_files = sort(filter(
            f -> occursin(r"m_sample_\d+\.(parquet|arrow)$", splitpath(f)[end]),
            readdir(m_sample_folder; join=true),
        ))
        if isempty(m_sample_files)
            _log("No existing M sample files found in $(m_sample_folder); creating fresh samples")
            effective_m_sample_count = isnothing(m_sample_count) ? max(1, min(num_k, max(8, min(32, processes_count * 2)))) : max(1, min(num_k, m_sample_count))
            create_m_samples(
                m_parquet_filename,
                output_folder,
                effective_m_sample_count,
                m_sample_size,
                Int.(iter_seeds[1:effective_m_sample_count]),
                start_year,
            )
            m_samples_dir = joinpath(output_folder, "m_samples")
            m_sample_files = [joinpath(m_samples_dir, "m_sample_$(lpad(i, 3, '0')).parquet") for i in 1:effective_m_sample_count]
        end
    else
        effective_m_sample_count = isnothing(m_sample_count) ? max(1, min(num_k, max(8, min(32, processes_count * 2)))) : max(1, min(num_k, m_sample_count))
        _log("Creating $(effective_m_sample_count) reusable M samples of $(m_sample_size) rows")
        create_m_samples(
            m_parquet_filename,
            output_folder,
            effective_m_sample_count,
            m_sample_size,
            Int.(iter_seeds[1:effective_m_sample_count]),
            start_year,
        )
        m_samples_dir = joinpath(output_folder, "m_samples")
        m_sample_files = [joinpath(m_samples_dir, "m_sample_$(lpad(i, 3, '0')).parquet") for i in 1:effective_m_sample_count]
    end

    map_args = Tuple{String, String, Int, Int}[]
    skipped = 0
    for (i, kf) in enumerate(k_files)
        if k_outputs_exist(kf, output_folder)
            skipped += 1
            continue
        end
        msf = m_sample_files[mod1(i, length(m_sample_files))]
        push!(map_args, (msf, kf, Int(iter_seeds[i]), Int(shuf_seeds[i])))
    end

    if skipped > 0
        _log("Skipped $(skipped) K grids because outputs already exist")
    end

    processed = String[]
    worker_ids = Int[]
    if !isempty(map_args) && processes_count > 1
        n_workers = min(processes_count, length(map_args))
        try
            worker_ids = addprocs(n_workers)
            this_file = @__FILE__
            @everywhere worker_ids begin
                if !isdefined(Main, :find_proportion_iteration)
                    include($this_file)
                end
            end
            _log("Using multiprocessing with $(length(worker_ids)) Julia workers")
        catch err
            @warn "Failed to initialize Julia workers; falling back to sequential execution" exception=(err, catch_backtrace())
            worker_ids = Int[]
        end
    end

    try
        total_batches = isempty(map_args) ? 0 : cld(length(map_args), batch_size)
        for b in 1:total_batches
            i1 = (b - 1) * batch_size + 1
            i2 = min(length(map_args), b * batch_size)
            batch = map_args[i1:i2]
            _log("Batch $(b)/$(total_batches)")

            if !isempty(worker_ids)
                batch_args = [
                    (
                        msf,
                        start_year,
                        evaluation_year,
                        output_folder,
                        kf,
                        iseed,
                        sseed,
                        min_potential_matches,
                        max_potential_matches,
                        similarity_mode,
                        propensity_caliper,
                        mahalanobis_lambda,
                        mahalanobis_bandwidth,
                        cluster_k_first,
                    )
                    for (msf, kf, iseed, sseed) in batch
                ]
                results = pmap(_iteration_tuple, batch_args)
                append!(processed, results)
            else
                for (msf, kf, iseed, sseed) in batch
                    kid = find_proportion_iteration(
                        msf,
                        start_year,
                        evaluation_year,
                        output_folder,
                        kf,
                        iseed,
                        sseed,
                        min_potential_matches,
                        max_potential_matches,
                        similarity_mode,
                        propensity_caliper,
                        mahalanobis_lambda,
                        mahalanobis_bandwidth,
                        cluster_k_first,
                    )
                    push!(processed, kid)
                end
            end

            _log("Processed $(length(processed))/$(num_k)")
        end
    finally
        if !isempty(worker_ids)
            rmprocs(worker_ids)
            _log("Closed $(length(worker_ids)) Julia workers")
        end
    end

    _log("Done. $(length(processed)) K grids processed.")

    if MEMORY_LOG_FILE !== nothing
        open(MEMORY_LOG_FILE, "a") do io
            println(io)
            println(io, "Completed: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
            println(io, "Total K grids: $(length(processed))")
        end
    end
end

function main(args::Vector{String})
    opts = parse_cli_args(args)

    k_directory = require_arg(opts, "--k_directory")
    m_parquet_filename = require_arg(opts, "--m_parquet_filename")
    output_folder = require_arg(opts, "--output_folder")
    start_year = parse(Int, require_arg(opts, "--start_year"))
    evaluation_year = parse(Int, require_arg(opts, "--evaluation_year"))

    m_sample_folder = parse_str_opt(opts, "--m_sample_folder", nothing)
    batch_size = parse_int_opt(opts, "--batch_size", 16)
    processes_count = parse_int_opt(opts, "--processes_count", 16)
    seed = parse_int_opt(opts, "--seed", 42)
    min_potential_matches = parse_int_opt(opts, "--min_potential_matches", 10)
    max_potential_matches = parse_int_opt(opts, "--max_potential_matches", 100)
    m_sample_count = haskey(opts, "--m_sample_count") ? parse(Int, opts["--m_sample_count"]) : nothing
    m_sample_size = parse_int_opt(opts, "--m_sample_size", 2_000_000)
    similarity_mode = parse_str_opt(opts, "--similarity_mode", parse_str_opt(opts, "--weight_mode", "hybrid_mahal_prop"))
    propensity_caliper = parse_float_opt(opts, "--propensity_caliper", 0.05)
    mahalanobis_lambda = parse_float_opt(opts, "--mahalanobis_lambda", 0.8)
    mahalanobis_bandwidth = parse_float_opt(opts, "--mahalanobis_bandwidth", 0.5)
    cluster_k_first = parse_int_opt(opts, "--cluster_k_first", 1) != 0

    find_pairs(
        k_directory,
        m_parquet_filename,
        m_sample_folder,
        start_year,
        evaluation_year,
        seed,
        output_folder,
        batch_size,
        processes_count,
        min_potential_matches,
        max_potential_matches,
        m_sample_count,
        m_sample_size,
        similarity_mode,
        propensity_caliper,
        mahalanobis_lambda,
        mahalanobis_bandwidth,
        cluster_k_first,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end