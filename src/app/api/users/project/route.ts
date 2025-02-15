import { connect } from "@/dbConfig/dbConfig"; // Database connection
import User from "@/models/userModel"; // User model
import { NextRequest, NextResponse } from "next/server"; // Next.js API response types
import { getDataFromToken } from "@/helpers/getDataFromToken"; // Helper to extract user data from token

// Establish database connection
connect();

// Define POST request handler to add a lecture
export async function POST(request: NextRequest) {
    try {
        // Extract userId (or email) from the token
        const userId = await getDataFromToken(request); // Get userId from the token
        if (!userId) {
            return NextResponse.json({ error: "User not authenticated" }, { status: 401 });
        }

        // Extract the topic from the request body
        const { topic } = await request.json();

        // Ensure topic is provided
        if (!topic) {
            return NextResponse.json({ error: "Project topic is required" }, { status: 400 });
        }

        // Find the user by userId (use _id to ensure you're querying correctly in MongoDB)
        const user = await User.findById(userId);
        if (!user) {
            return NextResponse.json({ error: "User does not exist" }, { status: 400 });
        }


        // Ensure the `lectures` field is initialized as an array
        if (!Array.isArray(user.projects)) {
            user.projects = []; // Initialize as an empty array if not already an array
        }

      

        // Add the new lecture topic to the user's lectures array
        user.projects.push({ topic });

        

        // Save the updated user data in the database
        const updatedUser = await user.save();

       

        return NextResponse.json({
            message: "Project added successfully",
            success: true,
        });
    } catch (error: any) {
        console.error("Error:", error); // Log any errors for debugging
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
