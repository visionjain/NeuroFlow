import { NextRequest } from "next/server";
import { spawn } from "child_process";
import path from "path";

export async function GET(req: NextRequest) {
    try {
        const scriptPath = path.join(process.cwd(), "Scripts", "linearreg.py");

        return new Response(
            new ReadableStream({
                start(controller) {
                    const process = spawn("python", ["-u", scriptPath]); // Unbuffered mode

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

                    // Handle standard output
                    process.stdout.on("data", (data) => sendData(data.toString()));
                    process.stderr.on("data", (data) => sendData(data.toString())); // No "ERROR:" prefix

                    // Process exit handling
                    process.on("close", (code) => {
                        if (!isClosed) {
                            sendData(`Process exited with code ${code}`);
                            sendData("END_OF_STREAM"); // Notify frontend that script has finished
                            controller.close();
                            isClosed = true;
                        }
                    });

                    // Handle unexpected errors
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
        // Send the error message via SSE format
        const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
        return new Response(`data: ${errorMessage}\n\n`, { status: 500 });
    }
}
