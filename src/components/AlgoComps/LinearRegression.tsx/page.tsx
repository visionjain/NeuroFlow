"use client";

import React, { useState, useRef } from "react";
import { FaPlay } from "react-icons/fa";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import axios from "axios";

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
    const [logs, setLogs] = useState<string>("");


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
        const eventSource = new EventSource("/api/users/scripts/linearregression");

        eventSource.onmessage = (event) => {
            setLogs((prevLogs) => prevLogs + event.data + "\n");
        };

        eventSource.onerror = (error) => {
            console.error("EventSource failed:", error);
            eventSource.close();
        };

        eventSource.addEventListener("end", () => {
            console.log("SSE Connection Closed.");
            eventSource.close();
        });
    };


    return (
        <div>
            <div className="text-xl">
                {/* Tabs Wraps Everything Now */}
                <Tabs defaultValue="home">
                    {/* Project Title & Tabs in One Row */}
                    <div className="flex items-center justify-between px-4 pt-4">
                        <div className="font-bold">
                            <h1 className="italic text-2xl">
                                {projectName} - {projectAlgo}{" "}
                                <span className="text-base lowercase">{projectTime}</span>
                            </h1>
                        </div>

                        {/* Tabs Navigation */}
                        <TabsList className="flex w-[40%] text-black dark:text-white bg-[#e6e6e6] dark:bg-[#0F0F0F]">
                            <TabsTrigger
                                className="w-[25%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]"
                                value="home"
                            >
                                Home
                            </TabsTrigger>
                            <TabsTrigger
                                className="w-[25%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]"
                                value="terminal"
                            >
                                Terminal
                            </TabsTrigger>
                        </TabsList>


                        <Button className="rounded-xl" onClick={handleRunScript}>
                            <FaPlay />
                        </Button>

                    </div>

                    {/* Tabs Content (Stays Fixed in Place) */}
                    <div className="mt-4">
                        <TabsContent value="home">
                            <div className="border border-[rgb(61,68,77)] flex flex-col gap-6 dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-4">
                                {/* First Row */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-48 rounded-xl w-1/3 bg-white">
                                        <div className="flex justify-center items-center gap-8 mt-8">
                                            {/* Train Data Selection */}
                                            <div className="flex flex-col items-center">
                                                <input
                                                    type="file"
                                                    id="trainDataset"
                                                    accept=".csv, .xlsx"
                                                    ref={trainInputRef}
                                                    onChange={(e) => handleFileSelect(e, "Train")}
                                                    hidden
                                                />
                                                <Button
                                                    className="w-28 h-28 flex flex-col justify-center items-center border-2 border-dashed border-gray-500 rounded-lg 
                        hover:bg-gray-100 hover:text-black dark:text-black dark:hover:bg-gray-800 dark:hover:border-gray-300 transition group"
                                                    onClick={() => trainInputRef.current?.click()}
                                                >
                                                    {trainFile ? (
                                                        <>
                                                            <span className="text-lg font-bold dark:text-black dark:group-hover:text-white">Train Data</span>
                                                            <p className="text-[10px] text-gray-500 dark:text-grey-600 dark:group-hover:text-white mt-2">{trainFile}</p>
                                                        </>
                                                    ) : (
                                                        <>
                                                            <span className="text-4xl">+</span>
                                                            <p className="text-sm mt-2">Select <br /> Train Data</p>
                                                        </>
                                                    )}
                                                </Button>
                                            </div>

                                            {/* Test Data Selection */}
                                            {showTestUpload && (
                                                <div className="flex flex-col items-center">
                                                    <input
                                                        type="file"
                                                        id="testDataset"
                                                        accept=".csv, .xlsx"
                                                        ref={testInputRef}
                                                        onChange={(e) => handleFileSelect(e, "Test")}
                                                        hidden
                                                    />
                                                    <Button
                                                        className="w-28 h-28 flex flex-col justify-center items-center border-2 border-dashed border-gray-500 rounded-lg 
                                               hover:bg-gray-100 hover:text-black dark:hover:bg-gray-800 dark:hover:border-gray-300 transition group"
                                                        onClick={() => testInputRef.current?.click()}
                                                    >
                                                        {testFile ? (
                                                            <>
                                                                <span className="text-lg font-bold dark:text-black dark:group-hover:text-white">Test Data</span>
                                                                <p className="text-[10px] text-gray-500 dark:text-grey-600 dark:group-hover:text-white mt-2">{testFile}</p>
                                                            </>
                                                        ) : (
                                                            <>
                                                                <span className="text-4xl">+</span>
                                                                <p className="text-sm mt-2">Select <br /> Test Data</p>
                                                            </>
                                                        )}
                                                    </Button>
                                                </div>
                                            )}
                                        </div>
                                        <p className="underline mt-2 flex justify-center text-sm text-blue-600 cursor-pointer" onClick={toggleTestDataset}>
                                            {showTestUpload ? "Don't have a test dataset?" : "Have a test dataset?"}
                                        </p>
                                    </div>
                                    <div className="dark:bg-[#212628] h-48 rounded-xl w-1/3 bg-white">
                                        <div>Select Train Column</div>
                                        <div>
                                            <Checkbox /> Column Names
                                        </div>
                                    </div>
                                    <div className="dark:bg-[#212628] h-48 rounded-xl w-1/3 bg-white">
                                        <div>Select Test Column</div>
                                        <div>
                                            <Checkbox /> Column Names
                                        </div>
                                    </div>
                                </div>

                                {/* Second Row */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-60 rounded-xl w-1/3 bg-white"></div>
                                    <div className="dark:bg-[#212628] h-60 rounded-xl w-1/3 bg-white"></div>
                                    <div className="dark:bg-[#212628] h-60 rounded-xl w-1/3 bg-white"></div>
                                </div>
                            </div>

                        </TabsContent>

                        {/* Terminal Tab Content */}
                        <TabsContent value="terminal">
                            <div className="border border-[rgb(61,68,77)] h-[490px] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 text-sm p-4 overflow-y-auto">
                                <pre className="whitespace-pre-wrap">{logs || "Terminal Output will be here."}</pre>
                            </div>
                        </TabsContent>
                    </div>
                </Tabs>
            </div>
        </div>
    );
};

export default LinearRegressionComponent;
