import { NextRequest } from "next/server";
import { spawn } from "child_process";
import path from "path";

export async function GET(req: NextRequest) {
  try {
    const scriptPath = path.join(process.cwd(), "Scripts", "linearreg.py");

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
    // Outlier Detection Parameters
    const enable_outlier_detection = searchParams.get("enable_outlier_detection") === "true";
    const outlier_method = searchParams.get("outlier_method");
    const z_score_threshold = searchParams.get("z_score_threshold");
    const iqr_lower = searchParams.get("iqr_lower");
    const iqr_upper = searchParams.get("iqr_upper");
    const winsor_lower = searchParams.get("winsor_lower");
    const winsor_upper = searchParams.get("winsor_upper");

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

    // Add Outlier Detection Parameters if enabled
    if (enable_outlier_detection) {
      args.push("--enable_outlier_detection", "true");
      args.push("--outlier_method", outlier_method || "");

      if (outlier_method === "Z-score" && z_score_threshold) {
        args.push("--z_score_threshold", z_score_threshold);
      }

      if (outlier_method === "IQR" && iqr_lower && iqr_upper) {
        args.push("--iqr_lower", iqr_lower);
        args.push("--iqr_upper", iqr_upper);
      }

      if (outlier_method === "Winsorization" && winsor_lower && winsor_upper) {
        args.push("--winsor_lower", winsor_lower);
        args.push("--winsor_upper", winsor_upper);
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
