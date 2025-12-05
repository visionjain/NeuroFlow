import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";

// GET endpoint to retrieve categorical values for dropdowns
export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const model_path = searchParams.get('model_path');

    if (!model_path) {
      return NextResponse.json(
        { error: "Missing model_path parameter" },
        { status: 400 }
      );
    }

    const model_dir = path.dirname(model_path);
    const preprocessing_path = path.join(model_dir, "preprocessing.pkl");

    if (!fs.existsSync(preprocessing_path)) {
      return NextResponse.json(
        { error: "Preprocessing file not found. Please retrain the model." },
        { status: 404 }
      );
    }

    // Use Python to read the pickle file
    const readScript = path.join(process.cwd(), "Scripts", "read_preprocessing.py");

    return new Promise((resolve) => {
      const process = spawn("python", ["-u", readScript, preprocessing_path]);

      let output = "";
      let errorOutput = "";

      process.stdout.on("data", (data) => {
        output += data.toString();
      });

      process.stderr.on("data", (data) => {
        errorOutput += data.toString();
      });

      process.on("close", (code) => {
        if (code !== 0) {
          resolve(
            NextResponse.json(
              { error: errorOutput || "Failed to read preprocessing info" },
              { status: 500 }
            )
          );
        } else {
          try {
            const data = JSON.parse(output);
            resolve(NextResponse.json(data));
          } catch (error) {
            resolve(
              NextResponse.json(
                { error: "Failed to parse preprocessing info" },
                { status: 500 }
              )
            );
          }
        }
      });
    });
  } catch (error: unknown) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error occurred";
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { model_path, input_values } = body;

    if (!model_path || !input_values) {
      return NextResponse.json(
        { error: "Missing required parameters (model_path, input_values)" },
        { status: 400 }
      );
    }

    // input_values should be an object like {feature1: value1, feature2: value2, ...}
    const predictScript = path.join(process.cwd(), "Scripts", "predict.py");

    return new Promise((resolve) => {
      const process = spawn("python", [
        "-u",
        predictScript,
        "--model_path",
        model_path,
        "--input_values",
        JSON.stringify(input_values),
      ]);

      let output = "";
      let errorOutput = "";

      process.stdout.on("data", (data) => {
        output += data.toString();
      });

      process.stderr.on("data", (data) => {
        errorOutput += data.toString();
      });

      process.on("close", (code) => {
        if (code !== 0) {
          resolve(
            NextResponse.json(
              { error: errorOutput || "Prediction failed" },
              { status: 500 }
            )
          );
        } else {
          try {
            // Parse the prediction from output
            const predictionMatch = output.match(/PREDICTION:(.+)/);
            const isBinaryMatch = errorOutput.match(/IS_BINARY:(True|False)/);
            
            if (predictionMatch) {
              const predictionStr = predictionMatch[1].trim();
              const isBinary = isBinaryMatch ? isBinaryMatch[1] === 'True' : null;
              
              // Try to parse as number, if it fails, keep as string (categorical prediction)
              const parsedNumber = parseFloat(predictionStr);
              const prediction = isNaN(parsedNumber) ? predictionStr : parsedNumber.toFixed(4);
              
              resolve(
                NextResponse.json({
                  prediction: prediction,
                  is_binary: isBinary,
                  raw_output: output,
                })
              );
            } else {
              resolve(
                NextResponse.json(
                  { error: "Could not parse prediction from output", output, errorOutput },
                  { status: 500 }
                )
              );
            }
          } catch (error) {
            resolve(
              NextResponse.json(
                { error: "Error parsing prediction result" },
                { status: 500 }
              )
            );
          }
        }
      });
    });
  } catch (error: unknown) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error occurred";
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}
