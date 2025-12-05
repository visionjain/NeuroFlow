import { connect } from "@/dbConfig/dbConfig";
import User from "@/models/userModel";
import { NextRequest, NextResponse } from "next/server";
import bcryptjs from "bcryptjs";

connect();

export async function POST(request: NextRequest) {
    try {
        const reqBody = await request.json();
        const { email, otp, newPassword } = reqBody;

        // Validate inputs
        if (!email || !otp || !newPassword) {
            return NextResponse.json({ error: "All fields are required" }, { status: 400 });
        }

        // Validate password strength
        if (newPassword.length < 6) {
            return NextResponse.json({ error: "Password must be at least 6 characters long" }, { status: 400 });
        }

        // Find user
        const user = await User.findOne({ email });
        if (!user) {
            return NextResponse.json({ error: "User not found" }, { status: 404 });
        }

        // Verify OTP again
        if (!user.forgotPasswordToken || user.forgotPasswordToken !== otp) {
            return NextResponse.json({ error: "Invalid OTP" }, { status: 400 });
        }

        // Check if OTP has expired
        if (user.forgotPasswordTokenExpiry < new Date()) {
            return NextResponse.json({ error: "OTP has expired" }, { status: 400 });
        }

        // Hash new password
        const hashedPassword = await bcryptjs.hash(newPassword, 10);

        // Update password and clear OTP fields
        user.password = hashedPassword;
        user.forgotPasswordToken = undefined;
        user.forgotPasswordTokenExpiry = undefined;
        await user.save();

        return NextResponse.json({
            message: "Password reset successfully",
            success: true
        });

    } catch (error: any) {
        console.error("Reset password error:", error);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
