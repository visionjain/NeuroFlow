import { connect } from "@/dbConfig/dbConfig"; // Database connection
import User from "@/models/userModel"; // User model
import { NextRequest, NextResponse } from "next/server"; // Next.js API response types
import { getDataFromToken } from "@/helpers/getDataFromToken"; // Helper to extract user data from token

// Establish database connection
connect();

export async function GET(request: NextRequest) {
  try {
    // Extract userId from the token
    const userId = await getDataFromToken(request);

    if (!userId) {
      return NextResponse.json({ error: "User not authenticated" }, { status: 401 });
    }

    // Find the user by userId
    const user = await User.findById(userId);
    if (!user) {
      return NextResponse.json({ error: "User does not exist" }, { status: 404 });
    }

    // Return the list of projects
    return NextResponse.json({
      projects: user.projects || [], // Return an empty array if no projects exist
      success: true,
    });
  } catch (error: any) {
    console.error("Error fetching projects:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
