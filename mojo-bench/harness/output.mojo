fn write_kernel_result_json(
    framework: String,
    framework_version: String,
    kernel_name: String,
    shape_name: String,
    mean_us: Float64,
    median_us: Float64,
    std_us: Float64,
    p5_us: Float64,
    p95_us: Float64,
    p99_us: Float64,
    ci95_lo: Float64,
    ci95_hi: Float64,
    n_iterations: Int,
    dw_statistic: Float64,
    tflops: Float64,
    gbs: Float64,
    roofline_pct: Float64,
    output_path: String,
):
    """Write benchmark result to JSON file.

    Uses framework-agnostic 'compilation' field per spec:
    {
      "compilation": {
        "mode": "ahead_of_time",
        "optimization": "--release",
        "graph_compilation": "N/A"
      }
    }
    """
    # Build JSON string manually (Mojo lacks serde)
    var json = String('{\n')
    json += '  "framework": "' + framework + '",\n'
    json += '  "framework_version": "' + framework_version + '",\n'
    json += '  "compilation": {\n'
    json += '    "mode": "ahead_of_time",\n'
    json += '    "optimization": "--release",\n'
    json += '    "graph_compilation": "N/A"\n'
    json += '  },\n'
    json += '  "kernel": "' + kernel_name + '",\n'
    json += '  "shape": "' + shape_name + '",\n'
    json += '  "stats": {\n'
    json += '    "mean_us": ' + str(mean_us) + ',\n'
    json += '    "median_us": ' + str(median_us) + ',\n'
    json += '    "std_us": ' + str(std_us) + ',\n'
    json += '    "p5_us": ' + str(p5_us) + ',\n'
    json += '    "p95_us": ' + str(p95_us) + ',\n'
    json += '    "p99_us": ' + str(p99_us) + ',\n'
    json += '    "ci95_lo": ' + str(ci95_lo) + ',\n'
    json += '    "ci95_hi": ' + str(ci95_hi) + ',\n'
    json += '    "n_iterations": ' + str(n_iterations) + ',\n'
    json += '    "dw_statistic": ' + str(dw_statistic) + '\n'
    json += '  },\n'
    json += '  "throughput": {\n'
    json += '    "tflops": ' + str(tflops) + ',\n'
    json += '    "gbs": ' + str(gbs) + ',\n'
    json += '    "roofline_pct": ' + str(roofline_pct) + '\n'
    json += '  }\n'
    json += '}\n'

    try:
        with open(output_path, "w") as f:
            f.write(json)
        print("Saved result to:", output_path)
    except:
        print("ERROR: Failed to write to:", output_path)
