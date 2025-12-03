import { connect } from "@/dbConfig/dbConfig";
import User from "@/models/userModel";
import { NextRequest, NextResponse } from "next/server";
import { getDataFromToken } from "@/helpers/getDataFromToken";
import fs from "fs";
import path from "path";

connect();

// GET - Load project state
export async function GET(request: NextRequest) {
  try {
    const userId = await getDataFromToken(request);
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get("projectId");

    if (!projectId) {
      return NextResponse.json({ error: "Project ID is required" }, { status: 400 });
    }

    const user = await User.findOne({ _id: userId });
    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    const project = user.projects.id(projectId);
    if (!project) {
      return NextResponse.json({ error: "Project not found" }, { status: 404 });
    }

    // Debug logging
    console.log("🔍 Loading state for project:", projectId);
    console.log("Project state exists:", !!project.state);
    console.log("Model trained:", project.state?.modelTrained);

    // Check if project has saved state
    if (!project.state || !project.state.modelTrained) {
      console.log("ℹ️ No saved state found");
      return NextResponse.json({ 
        hasState: false,
        message: "No saved state found for this project"
      });
    }

    // Validate that files still exist
    let isCorrupted = false;
    const missingFiles: string[] = [];

    if (project.state.datasetPath) {
      // Check if train file exists
      if (project.state.trainFile) {
        const trainPath = path.join(project.state.datasetPath, project.state.trainFile);
        if (!fs.existsSync(trainPath)) {
          isCorrupted = true;
          missingFiles.push(project.state.trainFile);
        }
      }

      // Check if test file exists (if it was used)
      if (project.state.testFile) {
        const testPath = path.join(project.state.datasetPath, project.state.testFile);
        if (!fs.existsSync(testPath)) {
          isCorrupted = true;
          missingFiles.push(project.state.testFile);
        }
      }

      // Check if model directory exists
      const modelDirName = `linearregression-${project.state.trainFile?.split(".")[0]}`;
      const modelDir = path.join(project.state.datasetPath, modelDirName);
      if (!fs.existsSync(modelDir)) {
        isCorrupted = true;
        missingFiles.push(`${modelDirName} directory`);
      }
    }

    // Update corruption status in database if needed
    if (isCorrupted && !project.state.isCorrupted) {
      project.state.isCorrupted = true;
      await user.save();
    }

    return NextResponse.json({
      hasState: true,
      isCorrupted,
      missingFiles,
      state: project.state
    });

  } catch (error: any) {
    console.error("Error loading project state:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

// POST - Save project state
export async function POST(request: NextRequest) {
  try {
    const userId = await getDataFromToken(request);
    const body = await request.json();
    const { projectId, state } = body;

    console.log("💾 POST - Saving state for project:", projectId);
    console.log("User ID:", userId);

    if (!projectId || !state) {
      console.error("❌ Missing projectId or state");
      return NextResponse.json({ error: "Project ID and state are required" }, { status: 400 });
    }

    const user = await User.findOne({ _id: userId });
    if (!user) {
      console.error("❌ User not found:", userId);
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    const project = user.projects.id(projectId);
    if (!project) {
      console.error("❌ Project not found:", projectId);
      return NextResponse.json({ error: "Project not found" }, { status: 404 });
    }

    console.log("📝 Updating project state...");
    
    // Update project state
    project.state = {
      ...state,
      lastRunAt: new Date(),
      isCorrupted: false
    };

    await user.save();
    
    console.log("✅ State saved successfully!");

    return NextResponse.json({
      message: "Project state saved successfully",
      success: true
    });

  } catch (error: any) {
    console.error("Error saving project state:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

// DELETE - Clear project state
export async function DELETE(request: NextRequest) {
  try {
    const userId = await getDataFromToken(request);
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get("projectId");

    if (!projectId) {
      return NextResponse.json({ error: "Project ID is required" }, { status: 400 });
    }

    const user = await User.findOne({ _id: userId });
    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    const project = user.projects.id(projectId);
    if (!project) {
      return NextResponse.json({ error: "Project not found" }, { status: 404 });
    }

    // Clear project state
    project.state = undefined;
    await user.save();

    return NextResponse.json({
      message: "Project state cleared successfully",
      success: true
    });

  } catch (error: any) {
    console.error("Error clearing project state:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
