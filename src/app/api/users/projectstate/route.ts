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
    const warnings: string[] = [];

    if (project.state.datasetPath) {
      // Check if train file exists (non-critical - just warn)
      if (project.state.trainFile) {
        const trainPath = path.join(project.state.datasetPath, project.state.trainFile);
        if (!fs.existsSync(trainPath)) {
          warnings.push(`Dataset file: ${project.state.trainFile}`);
          console.warn(`⚠️ Dataset file missing: ${project.state.trainFile}`);
        }
      }

      // Check if test file exists (non-critical - just warn)
      if (project.state.testFile) {
        const testPath = path.join(project.state.datasetPath, project.state.testFile);
        if (!fs.existsSync(testPath)) {
          warnings.push(`Test file: ${project.state.testFile}`);
          console.warn(`⚠️ Test file missing: ${project.state.testFile}`);
        }
      }

      // Check if model directory exists - use correct algorithm prefix
      let algorithmPrefix = 'linearregression';
      if (project.algorithm === 'logistic') {
        algorithmPrefix = 'logistic';
      } else if (project.algorithm === 'knn') {
        algorithmPrefix = 'knn';
      } else if (project.algorithm === 'linear') {
        algorithmPrefix = 'linearregression';
      }
      
      const modelDirName = `${algorithmPrefix}-${project.state.trainFile?.split(".")[0]}`;
      const modelDir = path.join(project.state.datasetPath, modelDirName);
      
      if (!fs.existsSync(modelDir)) {
        isCorrupted = true;
        missingFiles.push(`${modelDirName} directory`);
        console.error(`❌ Model directory missing: ${modelDir}`);
      } else {
        // Validate CRITICAL model files inside the directory
        const criticalModelFiles = ["model.pkl", "scaler.pkl", "preprocessing.pkl"];
        const missingCriticalFiles: string[] = [];
        
        // Check all critical model files (ALL must exist)
        for (const modelFile of criticalModelFiles) {
          const modelFilePath = path.join(modelDir, modelFile);
          if (!fs.existsSync(modelFilePath)) {
            missingCriticalFiles.push(modelFile);
            isCorrupted = true;
          }
        }
        
        if (missingCriticalFiles.length > 0) {
          missingFiles.push(...missingCriticalFiles);
          console.error(`❌ Missing critical model files:`, missingCriticalFiles);
        }
        
        // Validate CV fold models if they should exist (ALL must exist)
        if (project.state.enableCV && project.state.cvFolds) {
          const folds = parseInt(project.state.cvFolds);
          const missingFoldModels: string[] = [];
          
          for (let i = 1; i <= folds; i++) {
            const foldModelPath = path.join(modelDir, `model_fold_${i}.pkl`);
            if (!fs.existsSync(foldModelPath)) {
              missingFoldModels.push(`model_fold_${i}.pkl`);
            }
          }
          
          // ANY missing CV fold model means corrupted
          if (missingFoldModels.length > 0) {
            isCorrupted = true;
            missingFiles.push(...missingFoldModels);
            console.error(`❌ Missing ${missingFoldModels.length} CV fold models:`, missingFoldModels);
          }
        }
      }

      // Validate graph files and filter out missing ones (non-critical)
      if (project.state.generatedGraphs && Array.isArray(project.state.generatedGraphs)) {
        const validGraphs: string[] = [];
        const missingGraphs: string[] = [];
        
        for (const graphPath of project.state.generatedGraphs) {
          if (fs.existsSync(graphPath)) {
            validGraphs.push(graphPath);
          } else {
            missingGraphs.push(path.basename(graphPath));
          }
        }
        
        // Update the state with only valid graphs
        if (missingGraphs.length > 0) {
          console.log(`⚠️ Found ${missingGraphs.length} missing graphs:`, missingGraphs);
          project.state.generatedGraphs = validGraphs;
          
          // Graphs are regenerable - just warn, don't mark as corrupted
          if (missingGraphs.length > 0) {
            missingFiles.push(`${missingGraphs.length} graph file(s) - can be regenerated`);
          }
        }
      }
      
      // Validate results and logs - CRITICAL for project state
      if (project.state.modelTrained) {
        if (!project.state.results || project.state.results.trim() === "") {
          console.error("❌ Model trained but no results data saved");
          isCorrupted = true;
          missingFiles.push("Results data");
          project.state.modelTrained = false;
        }
        
        if (!project.state.logs || project.state.logs.trim() === "") {
          console.error("❌ Model trained but no logs saved");
          isCorrupted = true;
          missingFiles.push("Training logs");
        }
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
      warnings,
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
