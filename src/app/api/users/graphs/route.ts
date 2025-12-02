import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const imagePath = searchParams.get("path");

    if (!imagePath) {
      return new NextResponse("Missing image path", { status: 400 });
    }

    // Security check: ensure the path exists and is an image
    if (!fs.existsSync(imagePath)) {
      return new NextResponse("Image not found", { status: 404 });
    }

    const ext = path.extname(imagePath).toLowerCase();
    if (ext !== ".png" && ext !== ".jpg" && ext !== ".jpeg") {
      return new NextResponse("Invalid file type", { status: 400 });
    }

    // Read the image file
    const imageBuffer = fs.readFileSync(imagePath);

    // Return the image with appropriate content type
    return new NextResponse(imageBuffer, {
      headers: {
        "Content-Type": "image/png",
        "Cache-Control": "public, max-age=31536000",
      },
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
    return new NextResponse(errorMessage, { status: 500 });
  }
}
