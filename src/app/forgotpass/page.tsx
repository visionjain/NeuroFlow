"use client";

import DarkModeButton from '@/components/darkmode/page';
import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

const Forgotpass = () => {
    const router = useRouter();
    const [step, setStep] = useState(1); // 1: Email, 2: OTP, 3: New Password
    const [email, setEmail] = useState('');
    const [otp, setOtp] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [resendDisabled, setResendDisabled] = useState(false);
    const [countdown, setCountdown] = useState(0);

    // Step 1: Send OTP
    const handleSendOTP = async (e: React.FormEvent) => {
        e.preventDefault();
        
        if (!email) {
            toast.error('Please enter your email');
            return;
        }

        setLoading(true);
        try {
            const response = await axios.post('/api/users/forgot-password', { email });
            
            if (response.data.success) {
                toast.success('OTP sent to your email!');
                setStep(2);
                startResendTimer();
            }
        } catch (error: any) {
            toast.error(error.response?.data?.error || 'Failed to send OTP');
        } finally {
            setLoading(false);
        }
    };

    // Step 2: Verify OTP
    const handleVerifyOTP = async (e: React.FormEvent) => {
        e.preventDefault();
        
        if (!otp || otp.length !== 6) {
            toast.error('Please enter a valid 6-digit OTP');
            return;
        }

        setLoading(true);
        try {
            const response = await axios.post('/api/users/verify-otp', { email, otp });
            
            if (response.data.success) {
                toast.success('OTP verified successfully!');
                setStep(3);
            }
        } catch (error: any) {
            toast.error(error.response?.data?.error || 'Invalid OTP');
        } finally {
            setLoading(false);
        }
    };

    // Step 3: Reset Password
    const handleResetPassword = async (e: React.FormEvent) => {
        e.preventDefault();
        
        if (!newPassword || !confirmPassword) {
            toast.error('Please fill all fields');
            return;
        }

        if (newPassword.length < 6) {
            toast.error('Password must be at least 6 characters');
            return;
        }

        if (newPassword !== confirmPassword) {
            toast.error('Passwords do not match');
            return;
        }

        setLoading(true);
        try {
            const response = await axios.post('/api/users/reset-password', {
                email,
                otp,
                newPassword
            });
            
            if (response.data.success) {
                toast.success('Password reset successfully!');
                setTimeout(() => {
                    router.push('/login');
                }, 1500);
            }
        } catch (error: any) {
            toast.error(error.response?.data?.error || 'Failed to reset password');
        } finally {
            setLoading(false);
        }
    };

    // Resend OTP
    const handleResendOTP = async () => {
        setLoading(true);
        try {
            const response = await axios.post('/api/users/forgot-password', { email });
            
            if (response.data.success) {
                toast.success('OTP resent to your email!');
                setOtp('');
                startResendTimer();
            }
        } catch (error: any) {
            toast.error(error.response?.data?.error || 'Failed to resend OTP');
        } finally {
            setLoading(false);
        }
    };

    // Timer for resend button
    const startResendTimer = () => {
        setResendDisabled(true);
        setCountdown(60);
        
        const timer = setInterval(() => {
            setCountdown((prev) => {
                if (prev <= 1) {
                    clearInterval(timer);
                    setResendDisabled(false);
                    return 0;
                }
                return prev - 1;
            });
        }, 1000);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
            <DarkModeButton />
            
            <div className="flex items-center justify-center min-h-screen p-4">
                <Card className="w-full max-w-md">
                    <CardHeader className="space-y-1">
                        <CardTitle className="text-2xl font-bold text-center">
                            Reset Password
                        </CardTitle>
                        <CardDescription className="text-center">
                            {step === 1 && "Enter your email to receive an OTP"}
                            {step === 2 && "Enter the 6-digit OTP sent to your email"}
                            {step === 3 && "Create a new password for your account"}
                        </CardDescription>
                    </CardHeader>
                    
                    <CardContent>
                        {/* Step 1: Email Input */}
                        {step === 1 && (
                            <form onSubmit={handleSendOTP} className="space-y-4">
                                <div className="space-y-2">
                                    <Label htmlFor="email">Email Address</Label>
                                    <Input
                                        id="email"
                                        type="email"
                                        placeholder="your@email.com"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        required
                                        disabled={loading}
                                    />
                                </div>
                                
                                <Button
                                    type="submit"
                                    className="w-full"
                                    disabled={loading}
                                >
                                    {loading ? 'Sending...' : 'Send OTP'}
                                </Button>

                                <div className="text-center">
                                    <Button
                                        type="button"
                                        variant="link"
                                        onClick={() => router.push('/login')}
                                        className="text-sm"
                                    >
                                        Back to Login
                                    </Button>
                                </div>
                            </form>
                        )}

                        {/* Step 2: OTP Verification */}
                        {step === 2 && (
                            <form onSubmit={handleVerifyOTP} className="space-y-4">
                                <div className="space-y-2">
                                    <Label htmlFor="otp">Enter OTP</Label>
                                    <Input
                                        id="otp"
                                        type="text"
                                        placeholder="000000"
                                        value={otp}
                                        onChange={(e) => {
                                            const value = e.target.value.replace(/\D/g, '').slice(0, 6);
                                            setOtp(value);
                                        }}
                                        maxLength={6}
                                        required
                                        disabled={loading}
                                        className="text-center text-2xl tracking-widest"
                                    />
                                    <p className="text-xs text-gray-500 text-center">
                                        OTP sent to {email}
                                    </p>
                                </div>
                                
                                <Button
                                    type="submit"
                                    className="w-full"
                                    disabled={loading || otp.length !== 6}
                                >
                                    {loading ? 'Verifying...' : 'Verify OTP'}
                                </Button>

                                <div className="text-center space-y-2">
                                    <Button
                                        type="button"
                                        variant="link"
                                        onClick={handleResendOTP}
                                        disabled={resendDisabled || loading}
                                        className="text-sm"
                                    >
                                        {resendDisabled ? `Resend OTP in ${countdown}s` : 'Resend OTP'}
                                    </Button>
                                    <br />
                                    <Button
                                        type="button"
                                        variant="link"
                                        onClick={() => setStep(1)}
                                        className="text-sm"
                                    >
                                        Change Email
                                    </Button>
                                </div>
                            </form>
                        )}

                        {/* Step 3: New Password */}
                        {step === 3 && (
                            <form onSubmit={handleResetPassword} className="space-y-4">
                                <div className="space-y-2">
                                    <Label htmlFor="newPassword">New Password</Label>
                                    <Input
                                        id="newPassword"
                                        type="password"
                                        placeholder="Enter new password"
                                        value={newPassword}
                                        onChange={(e) => setNewPassword(e.target.value)}
                                        required
                                        disabled={loading}
                                    />
                                </div>

                                <div className="space-y-2">
                                    <Label htmlFor="confirmPassword">Confirm Password</Label>
                                    <Input
                                        id="confirmPassword"
                                        type="password"
                                        placeholder="Confirm new password"
                                        value={confirmPassword}
                                        onChange={(e) => setConfirmPassword(e.target.value)}
                                        required
                                        disabled={loading}
                                    />
                                </div>
                                
                                <Button
                                    type="submit"
                                    className="w-full"
                                    disabled={loading}
                                >
                                    {loading ? 'Resetting...' : 'Reset Password'}
                                </Button>
                            </form>
                        )}

                        {/* Progress Indicator */}
                        <div className="flex justify-center gap-2 mt-6">
                            <div className={`h-2 w-8 rounded-full ${step >= 1 ? 'bg-blue-600' : 'bg-gray-300'}`} />
                            <div className={`h-2 w-8 rounded-full ${step >= 2 ? 'bg-blue-600' : 'bg-gray-300'}`} />
                            <div className={`h-2 w-8 rounded-full ${step >= 3 ? 'bg-blue-600' : 'bg-gray-300'}`} />
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};

export default Forgotpass;
