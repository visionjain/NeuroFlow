import { connect } from "@/dbConfig/dbConfig";
import User from "@/models/userModel";
import { NextRequest, NextResponse } from "next/server";

connect();

export async function POST(request: NextRequest) {
    try {
        const reqBody = await request.json();
        const { email, otp } = reqBody;

        // Validate inputs
        if (!email || !otp) {
            return NextResponse.json({ error: "Email and OTP are required" }, { status: 400 });
        }

        // Find user
        const user = await User.findOne({ email });
        if (!user) {
            return NextResponse.json({ error: "User not found" }, { status: 404 });
        }

        // Check if OTP exists
        if (!user.forgotPasswordToken) {
            return NextResponse.json({ error: "No OTP found. Please request a new one." }, { status: 400 });
        }

        // Check if OTP has expired
        if (user.forgotPasswordTokenExpiry < new Date()) {
            return NextResponse.json({ error: "OTP has expired. Please request a new one." }, { status: 400 });
        }

        // Verify OTP
        if (user.forgotPasswordToken !== otp) {
            return NextResponse.json({ error: "Invalid OTP" }, { status: 400 });
        }

        // OTP is valid - return success
        return NextResponse.json({
            message: "OTP verified successfully",
            success: true
        });

    } catch (error: any) {
        console.error("Verify OTP error:", error);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
