"use client";

import React, { useState, useEffect, useRef } from "react";
import { FaPlay, FaSpinner } from "react-icons/fa";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import axios from "axios";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";

interface LinearRegressionProps {
    projectName: string;
    projectAlgo: string;
    projectTime: string;
}

const LinearRegressionComponent: React.FC<LinearRegressionProps> = ({ projectName, projectAlgo, projectTime }) => {
    const [trainFile, setTrainFile] = useState<string | null>(null);
    const [testFile, setTestFile] = useState<string | null>(null);
    const [showTestUpload, setShowTestUpload] = useState(true);
    const trainInputRef = useRef<HTMLInputElement>(null);
    const testInputRef = useRef<HTMLInputElement>(null);
    const terminalRef = useRef<HTMLDivElement>(null);
    const [isRunning, setIsRunning] = useState<boolean>(false);
    const [logs, setLogs] = useState<string>("");
    const [testSplitRatio, setTestSplitRatio] = useState<string>(""); // No default value


    useEffect(() => {
        if (terminalRef.current) {
            terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
        }
    }, [logs]);



    const handleTestSplitChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        let value = event.target.value;

        // Ensure valid number format
        if (/^\d*(\.\d{0,2})?$/.test(value)) {
            // Convert to number and ensure it stays within range
            let numValue = parseFloat(value);
            if (!isNaN(numValue) && numValue >= 0.01 && numValue <= 0.99) {
                setTestSplitRatio(value);
            }
        }
    };



    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>, type: string) => {
        const file = event.target.files?.[0];
        if (file) {
            if (type === "Train") {
                setTrainFile(file.name);
            } else if (type === "Test") {
                setTestFile(file.name);
            }
        }
    };

    const toggleTestDataset = () => {
        setShowTestUpload((prev) => {
            setTrainFile(null);
            setTestFile(null);
            if (trainInputRef.current) trainInputRef.current.value = "";
            if (testInputRef.current) testInputRef.current.value = "";
            return !prev;
        });
    };

    const handleRunScript = () => {
        setLogs(""); // Clear previous logs
        setIsRunning(true); // Disable button

        const eventSource = new EventSource("/api/users/scripts/linearregression");

        eventSource.onmessage = (event) => {
            if (event.data === "END_OF_STREAM") {
                eventSource.close();
                setIsRunning(false); // Re-enable button when script ends
            } else {
                setLogs((prevLogs) => prevLogs + event.data + "\n");
            }
        };

        eventSource.onerror = (error) => {
            console.error("EventSource failed:", error);
            eventSource.close();
            setIsRunning(false); // Re-enable button in case of an error
        };
    };




    return (
        <div>
            <div className="text-xl">
                {/* Tabs Wraps Everything Now */}
                <Tabs defaultValue="home">
                    {/* Project Title & Tabs in One Row */}
                    <div className="flex items-center justify-between px-4 mt-2">
                        <div className="font-bold">
                            <h1 className="italic text-2xl">
                                {projectName} - {projectAlgo}{" "}
                                <span className="text-sm lowercase">{projectTime}</span>
                            </h1>
                        </div>

                        {/* Tabs Navigation */}
                        <TabsList className="flex w-[40%] text-black dark:text-white bg-[#e6e6e6] dark:bg-[#0F0F0F]">
                            <TabsTrigger
                                className="w-[30%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]"
                                value="home"
                            >
                                Home
                            </TabsTrigger>
                            <TabsTrigger
                                className="w-[30%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]"
                                value="graphs"
                            >
                                Graphs
                            </TabsTrigger>
                            <TabsTrigger
                                className="w-[30%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]"
                                value="analysis"
                            >
                                Analysis
                            </TabsTrigger>
                            <TabsTrigger
                                className="w-[30%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]"
                                value="result"
                            >
                                Results
                            </TabsTrigger>
                            <TabsTrigger
                                className="w-[30%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]"
                                value="terminal"
                            >
                                Terminal
                            </TabsTrigger>
                        </TabsList>


                        <Button className="rounded-xl" onClick={handleRunScript} disabled={isRunning}>
                            {isRunning ? <FaSpinner className="animate-spin" /> : <FaPlay />}
                        </Button>

                    </div>

                    {/* Tabs Content (Stays Fixed in Place) */}
                    <div className="mt-2">
                        <TabsContent value="home">
                            <div className="border border-[rgb(61,68,77)] flex flex-col gap-3 dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-4">
                                {/* First Row */}
                                <div className="flex gap-x-3">



                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4">
                                        {/* Dataset Directory Path Input */}
                                        <div className="mb-4 text-center">
                                            <Label className="text-sm font-semibold">Dataset Directory Path</Label>
                                            <Input
                                                type="text"
                                                placeholder="Ex: D:\datasets"
                                                className="mt-1 dark:bg-[#0F0F0F]"
                                            />
                                        </div>

                                        {/* Train Data Selection & Dynamic Test Handling */}
                                        <div className="flex justify-between mt-4">
                                            {/* Train Data Selection */}
                                            <div className="flex flex-col items-center">
                                                <Label className="text-sm font-semibold mb-1">Train Data</Label>
                                                <input
                                                    type="file"
                                                    id="trainDataset"
                                                    accept=".csv, .xlsx"
                                                    ref={trainInputRef}
                                                    onChange={(e) => handleFileSelect(e, "Train")}
                                                    hidden
                                                />
                                                <Button
                                                    className="w-52 h-12 flex justify-center items-center border-2 border-dashed border-gray-500 rounded-md 
                hover:bg-gray-100 hover:text-black dark:text-black dark:hover:bg-gray-800 dark:hover:border-gray-300 transition"
                                                    onClick={() => trainInputRef.current?.click()}
                                                >
                                                    {trainFile ? (
                                                        <span className="text-sm truncate w-full text-center">{trainFile}</span>
                                                    ) : (
                                                        <span className="text-3xl">+</span>
                                                    )}
                                                </Button>
                                            </div>

                                            {/* Conditional UI: Test File Upload OR Test Set Split Ratio */}
                                            {showTestUpload ? (
                                                // Show Test File Selection
                                                <div className="flex flex-col items-center">
                                                    <Label className="text-sm font-semibold mb-1">Test Data</Label>
                                                    <input
                                                        type="file"
                                                        id="testDataset"
                                                        accept=".csv, .xlsx"
                                                        ref={testInputRef}
                                                        onChange={(e) => handleFileSelect(e, "Test")}
                                                        hidden
                                                    />
                                                    <Button
                                                        className="w-52 h-12 flex justify-center items-center border-2 border-dashed border-gray-500 rounded-md 
                hover:bg-gray-100 hover:text-black dark:hover:bg-gray-800 dark:hover:border-gray-300 transition"
                                                        onClick={() => testInputRef.current?.click()}
                                                    >
                                                        {testFile ? (
                                                            <span className="text-sm truncate w-full text-center">{testFile}</span>
                                                        ) : (
                                                            <span className="text-3xl">+</span>
                                                        )}
                                                    </Button>
                                                </div>
                                            ) : (
                                                // Show Test Set Split Ratio Input
                                                <div className="flex flex-col items-center">
                                                    <Label className="text-sm font-semibold mb-1">Test Set Split Ratio</Label>
                                                    <Input
                                                        type="text"
                                                        value={testSplitRatio}
                                                        onChange={handleTestSplitChange}
                                                        placeholder="Ex: 0.2"
                                                        className="w-52 h-12 text-center dark:bg-[#0F0F0F] border border-gray-500 rounded-md"
                                                    />
                                                </div>
                                            )}
                                        </div>

                                        {/* Toggle Link */}
                                        <p className="underline mt-2 flex justify-center text-sm text-blue-600 cursor-pointer" onClick={toggleTestDataset}>
                                            {showTestUpload ? "Don't have a test dataset?" : "Have a test dataset?"}
                                        </p>
                                    </div>



                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white">
                                        <div>Select Train Column</div>
                                        <div>
                                            <Checkbox /> Column Names
                                        </div>
                                    </div>
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white">
                                        <div>Select Test Column</div>
                                        <div>
                                            <Checkbox /> Column Names
                                        </div>
                                    </div>
                                </div>

                                {/* Second Row */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-64 rounded-xl w-1/3 bg-white"></div>
                                    <div className="dark:bg-[#212628] h-64 rounded-xl w-1/3 bg-white"></div>
                                    <div className="dark:bg-[#212628] h-64 rounded-xl w-1/3 bg-white"></div>
                                </div>
                            </div>

                        </TabsContent>

                        {/* Terminal Tab Content */}
                        <TabsContent value="terminal">
                            <div
                                ref={terminalRef}
                                className="border border-[rgb(61,68,77)] h-[505px] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 text-sm p-4 overflow-y-auto"
                            >
                                <pre className="whitespace-pre-wrap">{logs || "Terminal Output will be shown here."}</pre>
                            </div>
                        </TabsContent>
                    </div>
                </Tabs>
            </div>
        </div>
    );
};

export default LinearRegressionComponent;
