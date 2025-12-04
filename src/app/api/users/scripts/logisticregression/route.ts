import { NextRequest } from "next/server";
import { spawn } from "child_process";
import path from "path";

export async function GET(req: NextRequest) {
  try {
    const scriptPath = path.join(process.cwd(), "Scripts", "logisticreg.py");

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
    const feature_scaling = searchParams.get("feature_scaling");
    const selected_explorations = searchParams.get("available_Explorations");

    // Logistic Regression specific parameters
    const solver = searchParams.get("solver");
    const penalty = searchParams.get("penalty");
    const c_value = searchParams.get("c_value");
    const max_iter = searchParams.get("max_iter");
    const random_seed = searchParams.get("random_seed");
    const l1_ratio = searchParams.get("l1_ratio");

    // Class imbalance parameters
    const enable_imbalance = searchParams.get("enable_imbalance") === "true";
    const imbalance_method = searchParams.get("imbalance_method");
    const class_weight = searchParams.get("class_weight");

    // Advanced options
    const probability_threshold = searchParams.get("probability_threshold");
    const use_stratified_split = searchParams.get("use_stratified_split") === "true";
    const multi_class_strategy = searchParams.get("multi_class_strategy");

    // Cross-Validation Parameters
    const enable_cv = searchParams.get("enable_cv") === "true";
    const cv_folds = searchParams.get("cv_folds");

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

    if (feature_scaling) {
      args.push("--feature_scaling", feature_scaling);
    }

    if (selected_explorations) {
      args.push("--selected_explorations", selected_explorations);
    }

    // Add Logistic Regression specific parameters
    if (solver) {
      args.push("--solver", solver);
    }
    if (penalty) {
      args.push("--penalty", penalty);
    }
    if (c_value) {
      args.push("--c_value", c_value);
    }
    if (max_iter) {
      args.push("--max_iter", max_iter);
    }
    if (random_seed) {
      args.push("--random_seed", random_seed);
    }
    if (l1_ratio && penalty === "elasticnet") {
      args.push("--l1_ratio", l1_ratio);
    }

    // Add class imbalance parameters
    if (enable_imbalance) {
      args.push("--enable_imbalance", "true");
      if (imbalance_method) {
        args.push("--imbalance_method", imbalance_method);
      }
      if (class_weight) {
        args.push("--class_weight", class_weight);
      }
    }

    // Add advanced options
    if (probability_threshold) {
      args.push("--probability_threshold", probability_threshold);
    }
    if (use_stratified_split) {
      args.push("--use_stratified_split", "true");
    }
    if (multi_class_strategy) {
      args.push("--multi_class_strategy", multi_class_strategy);
    }

    // Add Cross-Validation Parameters
    if (enable_cv) {
      args.push("--enable_cv", "true");
      if (cv_folds) {
        args.push("--cv_folds", cv_folds);
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
