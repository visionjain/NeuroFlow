import { connect } from "@/dbConfig/dbConfig";
import User from "@/models/userModel";
import { NextRequest, NextResponse } from "next/server";
import bcryptjs from "bcryptjs";

connect();

export async function POST(request: NextRequest) {
    try {
        const reqBody = await request.json();
        const { name, email, phoneNumber, password } = reqBody;

        // Check if user already exists
        const user = await User.findOne({ email });
        const userphone = await User.findOne({ phoneNumber });

        if (user) {
            return NextResponse.json({ error: "User Email already exists" }, { status: 400 });
        }
        if (userphone) {
            return NextResponse.json({ error: "User Phone Number already exists" }, { status: 400 });
        }

        // Hash password
        const salt = await bcryptjs.genSalt(10);
        const hashedPassword = await bcryptjs.hash(password, salt);

        // Create a new user and initialize the 'project' field as an empty array
        const newUser = new User({
            name,
            email,
            phoneNumber,
            password: hashedPassword,
            project: [], 
        });

        const savedUser = await newUser.save();

        console.log("User created successfully:", savedUser); // Debugging line

        return NextResponse.json({
            message: "User created successfully",
            success: true,
            savedUser
        });
    } catch (error: any) {
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
