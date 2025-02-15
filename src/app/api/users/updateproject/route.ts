import { connect } from "@/dbConfig/dbConfig"; // Database connection
import User from "@/models/userModel"; // User model
import { NextRequest, NextResponse } from "next/server"; // Next.js API response types
import { getDataFromToken } from "@/helpers/getDataFromToken"; // Helper to extract user data from token

// Establish database connection
connect();

// Define the project type
type Project = {
  _id: string;
  topic: string;
};

// Define PUT request handler to update a project
export async function PUT(request: NextRequest) {
    try {
        // Extract userId (or email) from the token
        const userId = await getDataFromToken(request); // Get userId from the token

        if (!userId) {
            return NextResponse.json({ error: "User not authenticated" }, { status: 401 });
        }

        // Extract projectId and the new topic from the request body
        const { projectId, newTopic } = await request.json();

        // Ensure both projectId and newTopic are provided
        if (!projectId || !newTopic) {
            return NextResponse.json({ error: "Project ID and new topic are required" }, { status: 400 });
        }

        // Find the user by userId (use _id to ensure you're querying correctly in MongoDB)
        const user = await User.findById(userId);
        if (!user) {
            return NextResponse.json({ error: "User does not exist" }, { status: 400 });
        }

        // Type the project field
        const projects: Project[] = user.projects;

        // Find the project to update by projectId
        const projectIndex = projects.findIndex((project: Project) => project._id.toString() === projectId);
        if (projectIndex === -1) {
            return NextResponse.json({ error: "Project not found" }, { status: 404 });
        }

        // Update the topic of the found project
        projects[projectIndex].topic = newTopic;

        // Save the updated user data in the database
        const updatedUser = await user.save();

        return NextResponse.json({
            message: "Project updated successfully",
            success: true,
            project: projects[projectIndex], // Returning the updated project
        });
    } catch (error: any) {
        console.error("Error:", error); // Log any errors for debugging
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
