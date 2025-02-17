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

        if (!train_csv_path) {
            return new Response("Missing required parameter: train_csv_path", { status: 400 });
        }

        if (!train_columns || !output_column) {
            return new Response("Missing required parameters: train_columns or output_column", { status: 400 });
        }

        // Prepare command arguments
        const args = ["-u", scriptPath, "--train_csv_path", train_csv_path, "--train_columns", train_columns, "--output_column", output_column];

        if (test_csv_path && test_csv_path !== "None") {
            args.push("--test_csv_path", test_csv_path);
        } else if (test_split_ratio) {
            args.push("--test_split_ratio", test_split_ratio);
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
