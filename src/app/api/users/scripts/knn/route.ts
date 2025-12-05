import { NextRequest } from "next/server";
import { spawn } from "child_process";
import path from "path";

export async function GET(req: NextRequest) {
  try {
    const scriptPath = path.join(process.cwd(), "Scripts", "knn.py");

    // Extract query parameters
    const { searchParams } = new URL(req.url);
    const train_csv_path = searchParams.get("train_csv_path");
    const test_csv_path = searchParams.get("test_csv_path");
    const test_split_ratio = searchParams.get("test_split_ratio");
    const train_columns = searchParams.get("train_columns");
    const output_column = searchParams.get("output_column");
    const selected_graphs = searchParams.get("selected_graphs");
    const selected_missingval_tech = searchParams.get("selected_missingval_tech");
    const remove_Duplicates = searchParams.get("remove_Duplicates") === "true";
    const encoding_method = searchParams.get("encoding_Method");

    // KNN-specific parameters
    const k_value = searchParams.get("k_value");
    const enable_auto_k = searchParams.get("enable_auto_k") === "true";
    const k_range_start = searchParams.get("k_range_start");
    const k_range_end = searchParams.get("k_range_end");
    const distance_metric = searchParams.get("distance_metric");
    const weights = searchParams.get("weights");
    const algorithm = searchParams.get("algorithm");
    const leaf_size = searchParams.get("leaf_size");
    const p_value = searchParams.get("p_value");

    // Outlier Detection Parameters
    const enable_outlier_detection = searchParams.get("enable_outlier_detection") === "true";
    const outlier_method = searchParams.get("outlier_method");
    const z_score_threshold = searchParams.get("z_score_threshold");

    // Feature Scaling
    const feature_scaling = searchParams.get("feature_scaling");

    // Selected Data Exploration Techniques
    const selected_explorations = searchParams.get("available_Explorations");

    // Effect Features for Comparison
    const effect_features = searchParams.get("effect_features");

    // Advanced options
    const enable_cv = searchParams.get("enable_cv") === "true";
    const cv_folds = searchParams.get("cv_folds");
    const enable_dim_reduction = searchParams.get("enable_dim_reduction") === "true";
    const dim_reduction_method = searchParams.get("dim_reduction_method");
    const n_components = searchParams.get("n_components");
    const enable_imbalance = searchParams.get("enable_imbalance") === "true";
    const imbalance_method = searchParams.get("imbalance_method");

    if (!train_csv_path) {
      return new Response("Missing required parameter: train_csv_path", { status: 400 });
    }

    if (!train_columns || !output_column) {
      return new Response("Missing required parameters: train_columns or output_column", { status: 400 });
    }

    // Prepare command arguments
    const args = [
      "-u",
      scriptPath,
      "--train_csv_path",
      train_csv_path,
      "--train_columns",
      train_columns,
      "--output_column",
      output_column,
    ];

    if (test_csv_path && test_csv_path !== "None") {
      args.push("--test_csv_path", test_csv_path);
    } else if (test_split_ratio) {
      args.push("--test_split_ratio", test_split_ratio);
    }

    if (selected_graphs) {
      args.push("--selected_graphs", selected_graphs);
    }

    if (selected_missingval_tech) {
      args.push("--selected_missingval_tech", selected_missingval_tech);
    }

    if (remove_Duplicates) {
      args.push("--remove_duplicates");
    }

    if (encoding_method) {
      args.push("--encoding_type", encoding_method);
    }

    // Add KNN-specific parameters
    if (k_value) {
      args.push("--k_value", k_value);
    }

    if (enable_auto_k) {
      args.push("--enable_auto_k", "true");
      if (k_range_start) {
        args.push("--k_range_start", k_range_start);
      }
      if (k_range_end) {
        args.push("--k_range_end", k_range_end);
      }
    }

    if (distance_metric) {
      args.push("--distance_metric", distance_metric);
    }

    if (weights) {
      args.push("--weights", weights);
    }

    if (algorithm) {
      args.push("--algorithm", algorithm);
    }

    if (leaf_size) {
      args.push("--leaf_size", leaf_size);
    }

    if (p_value) {
      args.push("--p_value", p_value);
    }

    // Add Feature Scaling (REQUIRED for KNN)
    if (feature_scaling) {
      args.push("--feature_scaling", feature_scaling);
    }

    // Add Data Exploration Techniques
    if (selected_explorations) {
      args.push("--selected_explorations", selected_explorations);
    }

    // Add Effect Features
    if (effect_features) {
      args.push("--effect_features", effect_features);
    }

    // Add Outlier Detection Parameters if enabled
    if (enable_outlier_detection) {
      args.push("--enable_outlier_detection", "true");
      if (outlier_method) {
        args.push("--outlier_method", outlier_method);
      }
      if (outlier_method === "Z-score" && z_score_threshold) {
        args.push("--z_score_threshold", z_score_threshold);
      }
    }

    // Add Cross-Validation Parameters
    if (enable_cv) {
      args.push("--enable_cv", "true");
      if (cv_folds) {
        args.push("--cv_folds", cv_folds);
      }
    }

    // Add Dimensionality Reduction Parameters
    if (enable_dim_reduction) {
      args.push("--enable_dim_reduction", "true");
      if (dim_reduction_method) {
        args.push("--dim_reduction_method", dim_reduction_method);
      }
      if (n_components) {
        args.push("--n_components", n_components);
      }
    }

    // Add Class Imbalance Parameters
    if (enable_imbalance) {
      args.push("--enable_imbalance", "true");
      if (imbalance_method) {
        args.push("--imbalance_method", imbalance_method);
      }
    }

    return new Response(
      new ReadableStream({
        start(controller) {
          const process = spawn("python", args);

          process.stdout.setEncoding("utf8");
          process.stderr.setEncoding("utf8");

          let isClosed = false;

          const sendData = (data: string) => {
            if (!isClosed) {
              data.split("\n").forEach((line) => {
                if (line.trim()) {
                  controller.enqueue(`data: ${line.trim()}\n\n`);
                }
              });
            }
          };

          process.stdout.on("data", (data) => sendData(data.toString()));
          process.stderr.on("data", (data) => sendData(data.toString()));

          process.on("close", (code) => {
            if (!isClosed) {
              sendData("END_OF_STREAM");
              controller.close();
              isClosed = true;
            }
          });

          process.on("error", (err) => {
            if (!isClosed) {
              sendData(`Error: ${err.message}`);
              sendData("END_OF_STREAM");
              controller.close();
              isClosed = true;
            }
          });
        },
      }),
      {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
          "Access-Control-Allow-Origin": "*",
        },
      }
    );
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
    return new Response(`data: ${errorMessage}\n\n`, { status: 500 });
  }
}
