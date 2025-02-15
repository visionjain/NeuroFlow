import { connect } from "@/dbConfig/dbConfig"; // Database connection
import User from "@/models/userModel"; // User model
import { NextRequest, NextResponse } from "next/server"; // Next.js API response types
import { getDataFromToken } from "@/helpers/getDataFromToken"; // Helper to extract user data from token

// Establish database connection
connect();

// Define POST request handler to add a project
export async function POST(request: NextRequest) {
    try {
        // Extract userId from the token
        const userId = await getDataFromToken(request);
        if (!userId) {
            return NextResponse.json({ error: "User not authenticated" }, { status: 401 });
        }

        // Extract topic and algorithm from the request body
        const { topic, algorithm } = await request.json();

        // Validate inputs
        if (!topic) {
            return NextResponse.json({ error: "Project topic is required" }, { status: 400 });
        }
        if (!algorithm) {
            return NextResponse.json({ error: "Algorithm is required" }, { status: 400 });
        }

        // Find the user by userId
        const user = await User.findById(userId);
        if (!user) {
            return NextResponse.json({ error: "User does not exist" }, { status: 404 });
        }

        // Ensure `projects` is an array
        if (!Array.isArray(user.projects)) {
            user.projects = []; 
        }

        // Add the new project as an object containing both topic and algorithm
        user.projects.push({ topic, algorithm });

        // Save the updated user data in the database
        await user.save();

        return NextResponse.json({
            message: "Project added successfully",
            success: true,
            project: { topic, algorithm }, // Return the added project
        });
    } catch (error: any) {
        console.error("Error:", error);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
