import { connect } from "@/dbConfig/dbConfig";
import User from "@/models/userModel";
import { NextRequest, NextResponse } from "next/server";
import nodemailer from 'nodemailer';

connect();

// Generate 6-digit OTP
function generateOTP(): string {
    return Math.floor(100000 + Math.random() * 900000).toString();
}

export async function POST(request: NextRequest) {
    try {
        const reqBody = await request.json();
        const { email } = reqBody;

        // Validate email
        if (!email) {
            return NextResponse.json({ error: "Email is required" }, { status: 400 });
        }

        // Find user by email
        const user = await User.findOne({ email });
        if (!user) {
            return NextResponse.json({ error: "No account found with this email" }, { status: 404 });
        }

        // Generate OTP
        const otp = generateOTP();
        
        // Store OTP in database with 10 minute expiry
        user.forgotPasswordToken = otp;
        user.forgotPasswordTokenExpiry = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes
        await user.save();

        // Setup email transporter
        const transport = nodemailer.createTransport({
            service: 'gmail',
            auth: {
                user: process.env.EMAIL_USER,
                pass: process.env.EMAIL_PASSWORD
            }
        });

        // Email content
        const mailOptions = {
            from: process.env.EMAIL_USER,
            to: email,
            subject: "Password Reset OTP - NeuroFlow",
            html: `
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <h2 style="color: #333;">Password Reset Request</h2>
                    <p>Hello ${user.name},</p>
                    <p>You have requested to reset your password. Please use the following OTP to proceed:</p>
                    <div style="background-color: #f4f4f4; padding: 20px; text-align: center; margin: 20px 0;">
                        <h1 style="color: #4CAF50; letter-spacing: 5px; margin: 0;">${otp}</h1>
                    </div>
                    <p><strong>This OTP will expire in 10 minutes.</strong></p>
                    <p>If you did not request a password reset, please ignore this email.</p>
                    <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;">
                    <p style="color: #666; font-size: 12px;">This is an automated email from NeuroFlow. Please do not reply.</p>
                </div>
            `
        };

        // Send email
        await transport.sendMail(mailOptions);

        return NextResponse.json({
            message: "OTP sent to your email successfully",
            success: true
        });

    } catch (error: any) {
        console.error("Forgot password error:", error);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}
