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

                    const sendData = (data: string) => {
                        data.split("\n").forEach((line) => {
                            if (line.trim()) controller.enqueue(`data: ${line.trim()}\n\n`);
                        });
                    };

                    // Stream both stdout and stderr
                    process.stdout.on("data", (data) => sendData(data.toString()));
                    process.stderr.on("data", (data) => sendData(data.toString()));

                    // Handle process closure
                    process.on("close", (code) => {
                        sendData(`Process exited with code ${code}`);
                        setTimeout(() => controller.close(), 100); // Ensure all logs are sent before closing
                    });

                    // Handle process errors
                    process.on("error", (err) => {
                        sendData(`Error: ${err.message}`);
                        setTimeout(() => controller.close(), 100);
                    });
                },
            }),
            {
                headers: {
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    Connection: "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                },
            }
        );
    } catch (error) {
        return new Response(`data: Error: ${error}\n\n`, { status: 500 });
    }
}
