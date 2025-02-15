import { connect } from "@/dbConfig/dbConfig"; // Database connection
import User from "@/models/userModel"; // User model
import { NextRequest, NextResponse } from "next/server"; // Next.js API response types
import { getDataFromToken } from "@/helpers/getDataFromToken"; // Helper to extract user data from token

// Define the project type
interface Project {
  _id: string; // MongoDB ObjectId as string
  topic: string;
}

// Establish database connection
connect();

// Define the DELETE request handler to delete a project
export async function DELETE(request: NextRequest) {
  try {
    const userId = await getDataFromToken(request); // Get user's _id from token

    if (!userId) {
      return NextResponse.json({ error: "User not authenticated" }, { status: 401 });
    }

    // Find the user by userId (_id)
    const user = await User.findById(userId); // Use _id to query the user
    if (!user) {
      return NextResponse.json({ error: "User does not exist" }, { status: 400 });
    }

    const userEmail = user.email; // Access the user's email from the found user object

    // Extract the projectId from the request body
    const { projectId } = await request.json();

    // Ensure the projectId is provided
    if (!projectId) {
      return NextResponse.json({ error: "Project ID is required" }, { status: 400 });
    }

    // Ensure the `projects` field is initialized as an array
    if (!Array.isArray(user.projects)) {
      user.projects = []; // Initialize as an empty array if not already an array
    }

    // Find the project to delete by projectId
    const projectIndex = user.projects.findIndex((project: Project) => project._id.toString() === projectId);
    if (projectIndex === -1) {
      return NextResponse.json({ error: "Project not found" }, { status: 404 });
    }

    // Remove the project from the user's project array
    user.projects.splice(projectIndex, 1);

    // Save the updated user data in the database
    await user.save();

    return NextResponse.json({
      message: "Project deleted successfully",
      success: true,
    });
  } catch (error: any) {
    console.error("Error:", error); // Log any errors for debugging
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
