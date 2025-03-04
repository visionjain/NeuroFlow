"use client";

import React, { useState, useEffect, useRef } from "react";
import { FaPlay, FaSpinner } from "react-icons/fa";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
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
    const [datasetPath, setDatasetPath] = useState<string>("");
    const [showTestUpload, setShowTestUpload] = useState(true);
    const trainInputRef = useRef<HTMLInputElement>(null);
    const testInputRef = useRef<HTMLInputElement>(null);
    const terminalRef = useRef<HTMLDivElement>(null);
    const [isRunning, setIsRunning] = useState<boolean>(false);
    const [logs, setLogs] = useState<string>("");
    const [testSplitRatio, setTestSplitRatio] = useState<string>("0.2");
    const [trainColumns, setTrainColumns] = useState<string[]>([]);
    const [selectedTrainColumns, setSelectedTrainColumns] = useState<string[]>([]);
    const [selectedOutputColumn, setSelectedOutputColumn] = useState<string | null>(null);
    const [results, setResults] = useState<string>("");
    const [selectedGraphs, setSelectedGraphs] = useState<string[]>([]);
    const [selectedHandlingMissingValue, setSelectedHandlingMissingValue] = useState<string>("Drop Rows with Missing Values");
    const [removeDuplicates, setRemoveDuplicates] = useState(true);
    const [enableOutlierDetection, setEnableOutlierDetection] = useState(false);
    const [outlierMethod, setOutlierMethod] = useState(""); // Selected method
    const [zScoreThreshold, setZScoreThreshold] = useState(3.0);
    const [iqrLower, setIqrLower] = useState(1.5);
    const [iqrUpper, setIqrUpper] = useState(1.5);
    const [winsorLower, setWinsorLower] = useState(1);
    const [winsorUpper, setWinsorUpper] = useState(99);
    const [encodingMethod, setEncodingMethod] = useState("one-hot");
    const [selectedExplorations, setSelectedExplorations] = useState<string[]>([]);



    const availableExplorations = [
        "First 5 Rows",
        "Last 5 Rows",
        "Dataset Shape",
        "Data Types",
        "Summary Statistics",
        "Missing Values",
        "Unique Values Per Column",
        "Duplicate Rows",
        "Min & Max Values",
        "Correlation Matrix",
        "Skewness",
        "Target Column Distribution"
    ];


    const availableGraphs = [
        "Heatmap",
        "Histogram Distribution",
        "Histogram Residuals",
        "Individual Effect Plot",
        "Mean Effect Plot",
        "Model Coefficients",
        "Residual Plot",
        "Shap Summary Plot",
        "Trend Effect Plot",
        "Box Plot",
    ];
    const availableHandlingMissingValues = [
        "Mean Imputation",
        "Median Imputation",
        "Mode Imputation",
        "Forward/Backward Fill",
        "Drop Rows with Missing Values",
    ];


    useEffect(() => {
        if (terminalRef.current) {
            terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
        }
    }, [logs]);



    const handleTestSplitChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        let value = event.target.value;

        // Allow only valid decimal numbers (empty string is also allowed for smooth typing)
        if (/^\d*\.?\d{0,2}$/.test(value) || value === "") {
            setTestSplitRatio(value);
        }
    };
    const handleTestSplitBlur = () => {
        let numValue = parseFloat(testSplitRatio);
        if (isNaN(numValue) || numValue < 0.01 || numValue > 0.99) {
            setTestSplitRatio("0.2"); // Default value
        }
    };


    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>, type: string) => {
        const file = event.target.files?.[0];
        if (file) {
            const fileName = file.name;

            if (type === "Train") {
                setTrainFile(fileName);

                // Read the first line of the CSV to get column names
                const reader = new FileReader();
                reader.onload = (e) => {
                    const result = e.target?.result as string;
                    if (result) {
                        const firstLine = result.split("\n")[0];
                        const columns = firstLine.split(",").map(col => col.trim()); // Handle spaces
                        setTrainColumns(columns);
                    }
                };
                reader.readAsText(file);
            } else if (type === "Test") {
                setTestFile(fileName);
            }
        }
    };

    // Toggle individual selection
    const toggleExploration = (technique: string) => {
        setSelectedExplorations((prev) =>
            prev.includes(technique) ? prev.filter((item) => item !== technique) : [...prev, technique]
        );
    };

    // Select all / Deselect all
    const toggleSelectAllExplorations = () => {
        if (selectedExplorations.length === availableExplorations.length) {
            setSelectedExplorations([]); // Deselect all
        } else {
            setSelectedExplorations(availableExplorations); // Select all
        }
    };


    const toggleTrainColumn = (column: string) => {
        setSelectedTrainColumns(prev =>
            prev.includes(column) ? prev.filter(col => col !== column) : [...prev, column]
        );
    };

    const handleOutputColumnSelect = (column: string) => {
        setSelectedOutputColumn(column);
    };






    const toggleTestDataset = () => {
        setShowTestUpload((prev) => {
            // Always clear the columns when the toggle is clicked
            setTrainColumns([]); // Clear the train columns
            setSelectedTrainColumns([]); // Reset the selected train columns
            setSelectedOutputColumn(null); // Reset the output column

            // Reset the file selections
            setTrainFile(null);
            setTestFile(null);

            // Reset the input fields for file selection
            if (trainInputRef.current) trainInputRef.current.value = "";
            if (testInputRef.current) testInputRef.current.value = "";

            return !prev;
        });
    };


    const toggleGraph = (graph: string) => {
        setSelectedGraphs((prev) =>
            prev.includes(graph)
                ? prev.filter((g) => g !== graph)
                : [...prev, graph]
        );
    };

    // Function to select/deselect all graphs
    const toggleSelectAllGraphs = () => {
        if (selectedGraphs.length === availableGraphs.length) {
            setSelectedGraphs([]);
        } else {
            setSelectedGraphs([...availableGraphs]);
        }
    };






    const handleRunScript = () => {
        setLogs(""); // Clear previous logs
        setResults(""); // Clear previous results
        setIsRunning(true); // Disable button

        if (!datasetPath || !trainFile) {
            alert("Dataset path and train file are required.");
            setIsRunning(false);
            return;
        }

        if (selectedTrainColumns.length === 0) {
            alert("Please select at least one train column.");
            setIsRunning(false);
            return;
        }

        if (!selectedOutputColumn) {
            alert("Please select an output column.");
            setIsRunning(false);
            return;
        }

        // Normalize datasetPath to remove trailing slashes
        let normalizedPath = datasetPath.trim();
        if (normalizedPath.endsWith("\\") || normalizedPath.endsWith("/")) {
            normalizedPath = normalizedPath.slice(0, -1);
        }

        // Detect OS to use appropriate separator (this example assumes Windows or Unix-like)
        const isWindows = navigator.platform.startsWith("Win");
        const separator = isWindows ? "\\\\" : "/";

        // Construct file paths
        const train_csv_path = `${normalizedPath}${separator}${trainFile}`;
        const test_csv_path = testFile ? `${normalizedPath}${separator}${testFile}` : "None";

        // Prepare API query parameters
        const queryParams = new URLSearchParams({
            train_csv_path,
            test_csv_path,
            train_columns: JSON.stringify(selectedTrainColumns),
            output_column: selectedOutputColumn,
            selected_graphs: JSON.stringify(selectedGraphs),
            selected_missingval_tech: JSON.stringify(selectedHandlingMissingValue),
            remove_Duplicates: JSON.stringify(removeDuplicates),
            encoding_Method: encodingMethod,
            available_Explorations: JSON.stringify(selectedExplorations),

            // Outlier Detection Parameters
            enable_outlier_detection: JSON.stringify(enableOutlierDetection),
            outlier_method: JSON.stringify(outlierMethod), // "Z-score" / "IQR" / "Winsorization"

            // Z-score Method (if selected)
            ...(outlierMethod === "Z-score" && {
                z_score_threshold: JSON.stringify(zScoreThreshold),
            }),

            // IQR Method (if selected)
            ...(outlierMethod === "IQR" && {
                iqr_lower: JSON.stringify(iqrLower),
                iqr_upper: JSON.stringify(iqrUpper),
            }),

            // Winsorization Method (if selected)
            ...(outlierMethod === "Winsorization" && {
                winsor_lower: JSON.stringify(winsorLower),
                winsor_upper: JSON.stringify(winsorUpper),
            }),
        });


        if (!testFile && testSplitRatio) {
            queryParams.append("test_split_ratio", testSplitRatio);
        }

        const apiUrl = `/api/users/scripts/linearregression?${queryParams.toString()}`;

        // Local variable to accumulate all output lines
        let allLogs = "";
        const eventSource = new EventSource(apiUrl);

        eventSource.onmessage = (event) => {
            if (event.data === "END_OF_STREAM") {
                // Once stream ends, parse accumulated logs for results lines
                const resultLines = allLogs
                    .split("\n")
                    .filter(
                        (line) =>
                            line.startsWith("Mean Squared Error:") ||
                            line.startsWith("R-squared Score:") ||
                            line.startsWith("Accuracy Score:") ||
                            line.includes("Skipping accuracy")
                    );
                setResults(resultLines.join("\n"));
                eventSource.close();
                setIsRunning(false);
            } else {
                allLogs += event.data + "\n";
                setLogs((prev) => prev + event.data + "\n");
            }
        };

        eventSource.onerror = (error) => {
            console.error("EventSource failed:", error);
            eventSource.close();
            setIsRunning(false);
        };
    };



    const toggleSelectAll = () => {
        if (selectedTrainColumns.length === trainColumns.length) {
            setSelectedTrainColumns([]); // deselect all
        } else {
            setSelectedTrainColumns([...trainColumns]); // select all
        }
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
                                                placeholder="Ex: D:\datasetpath"
                                                className="mt-1 dark:bg-[#0F0F0F]"
                                                value={datasetPath}
                                                onChange={(e) => setDatasetPath(e.target.value)}
                                            />

                                        </div>

                                        {/* Train Data Selection & Dynamic Test Handling */}
                                        <div className="flex w-full gap-2">
                                            {/* Train Data Upload */}
                                            <div className="flex flex-col w-full items-center">
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
                                                    className="h-12 w-full flex justify-center items-center border-2 border-dashed border-gray-500 rounded-md 
            hover:bg-gray-100 hover:text-black dark:hover:bg-gray-800 dark:hover:border-gray-300 transition"
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
                                            <div className="flex flex-col w-full items-center">


                                                {showTestUpload ? (

                                                    // Test File Selection
                                                    <>
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
                                                            className="h-12 w-full flex justify-center items-center border-2 border-dashed border-gray-500 rounded-md 
                    hover:bg-gray-100 hover:text-black dark:hover:bg-gray-800 dark:hover:border-gray-300 transition"
                                                            onClick={() => testInputRef.current?.click()}
                                                        >
                                                            {testFile ? (
                                                                <span className="text-sm truncate w-full text-center">{testFile}</span>
                                                            ) : (
                                                                <span className="text-3xl">+</span>
                                                            )}
                                                        </Button>
                                                    </>
                                                ) : (
                                                    // Test Set Split Ratio Input
                                                    <div className="flex flex-col w-full items-center">
                                                        <Label className="text-sm font-semibold mb-1">Test Set Split Ratio</Label>
                                                        <Input
                                                            type="text"
                                                            value={testSplitRatio}
                                                            onChange={handleTestSplitChange}
                                                            onBlur={handleTestSplitBlur}
                                                            placeholder="Ex: 0.2"
                                                            className="w-full h-12 text-center dark:bg-[#0F0F0F] border border-gray-500 rounded-md"
                                                        />
                                                    </div>
                                                )}
                                            </div>
                                        </div>



                                        {/* Toggle Link */}
                                        <p className="underline mt-2 flex justify-center text-sm text-blue-600 cursor-pointer" onClick={toggleTestDataset}>
                                            {showTestUpload ? "Don't have a test dataset?" : "Have a test dataset?"}
                                        </p>
                                    </div>


                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm">Select Train Columns</div>

                                            {/* Show "Select All" Checkbox only if a file is selected */}
                                            {trainFile && (
                                                <div className="flex items-center">
                                                    <Checkbox
                                                        checked={selectedTrainColumns.length === trainColumns.length}
                                                        onCheckedChange={() => toggleSelectAll()}
                                                    />
                                                    <span className="ml-1 text-xs">Select All</span>
                                                </div>
                                            )}
                                        </div>

                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            {trainFile ? (
                                                <div className="grid grid-cols-2 gap-1">
                                                    {trainColumns.map((col, index) => (
                                                        <div key={index} className="flex items-center text-xs">
                                                            <Checkbox
                                                                checked={selectedTrainColumns.includes(col)}
                                                                onCheckedChange={() => toggleTrainColumn(col)}
                                                            />
                                                            <span className="ml-1">{col}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            ) : (
                                                <div className="text-center">Please select a train file to enable column selection.</div>
                                            )}
                                        </div>
                                    </div>



                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-1 mt-1">Select Output Column</div>

                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            {trainFile ? (
                                                <div className="grid grid-cols-2 gap-1">
                                                    {trainColumns.map((col, index) => (
                                                        <div key={index} className="flex items-center text-xs">
                                                            <Checkbox
                                                                checked={selectedOutputColumn === col}
                                                                onCheckedChange={() => handleOutputColumnSelect(col)}
                                                            />
                                                            <span className="ml-1">{col}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            ) : (
                                                <div className="text-center">Please select a train file to select output column.</div>
                                            )}
                                        </div>
                                    </div>



                                </div>

                                {/* Second Row */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm">Select Graphs</div>
                                            <div className="flex items-center">
                                                <Checkbox
                                                    checked={selectedGraphs.length === availableGraphs.length}
                                                    onCheckedChange={toggleSelectAllGraphs}
                                                />
                                                <span className="ml-2 text-xs">Select All</span>
                                            </div>
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            <div>
                                                {trainFile ? (
                                                    <>

                                                        <div className="grid grid-cols-2 gap-1">
                                                            {availableGraphs.map((graph) => (
                                                                <div key={graph} className="flex items-center text-xs">
                                                                    <Checkbox
                                                                        checked={selectedGraphs.includes(graph)}
                                                                        onCheckedChange={() => toggleGraph(graph)}
                                                                    />
                                                                    <span className="ml-1">{graph}</span>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </>
                                                ) : (
                                                    <div className="text-center">Please select a train file to enable graph selection.</div>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm">Handling Missing Values</div>

                                            {/* Remove Duplicates Checkbox (Only shows if a file is selected) */}
                                            {trainFile && (
                                                <div className="flex items-center text-xs cursor-pointer">

                                                    <Checkbox
                                                        checked={removeDuplicates}
                                                        onCheckedChange={(checked) => setRemoveDuplicates(!!checked)} // Ensures it's always a boolean
                                                        className="mr-2"
                                                    />
                                                    <span>Remove Duplicates</span>
                                                </div>
                                            )}
                                        </div>

                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            {trainFile ? (
                                                <div className="grid grid-cols-1 gap-1">
                                                    {availableHandlingMissingValues.map((method) => (
                                                        <label key={method} className="flex items-center text-xs cursor-pointer">
                                                            <input
                                                                type="radio"
                                                                name="missingValueHandling"
                                                                value={method}
                                                                checked={selectedHandlingMissingValue === method}
                                                                onChange={() => setSelectedHandlingMissingValue(method)}
                                                                className="mr-2"
                                                            />
                                                            <span>{method}</span>
                                                        </label>
                                                    ))}
                                                </div>
                                            ) : (
                                                <div className="text-center">
                                                    Please select a train file to enable options.
                                                </div>
                                            )}
                                        </div>
                                    </div>



                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm ">Outlier Removal</div>
                                            {trainFile && (
                                                <div className="flex items-center text-xs cursor-pointer">
                                                    <Checkbox
                                                        checked={enableOutlierDetection}
                                                        onCheckedChange={(checked) => setEnableOutlierDetection(!!checked)}
                                                        className="mr-2"
                                                    />
                                                    <span>Enable Outlier Removal</span>
                                                </div>
                                            )}
                                        </div>

                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            {trainFile ? (
                                                <>
                                                    {enableOutlierDetection ? (
                                                        <>
                                                            {/* Outlier Detection Method Selection */}
                                                            <Select onValueChange={setOutlierMethod} value={outlierMethod}>
                                                                <SelectTrigger className="w-full mt-3 text-xs text-white">
                                                                    <SelectValue placeholder="Select Outlier Method" />
                                                                </SelectTrigger>
                                                                <SelectContent>
                                                                    <SelectItem value="z-score">Z-Score Method</SelectItem>
                                                                    <SelectItem value="iqr">IQR Method</SelectItem>
                                                                    <SelectItem value="winsorization">Winsorization</SelectItem>
                                                                </SelectContent>
                                                            </Select>

                                                            {/* Method-Specific Inputs */}
                                                            {outlierMethod === "z-score" && (
                                                                <div className="mt-2">
                                                                    <span className="text-xs text-white">Z-Score Threshold:</span>
                                                                    <Input
                                                                        type="number"
                                                                        className="mt-1 w-full text-xs"
                                                                        value={zScoreThreshold}
                                                                        onChange={(e) => setZScoreThreshold(parseFloat(e.target.value))}
                                                                    />
                                                                </div>
                                                            )}

                                                            {outlierMethod === "iqr" && (
                                                                <div className="mt-2">
                                                                    <span className="text-xs text-white">IQR Multiplier (Lower bound & Upper bound):</span>
                                                                    <div className="flex space-x-2">
                                                                        <Input
                                                                            type="number"
                                                                            className="mt-1 w-1/2 text-xs"
                                                                            value={iqrLower}
                                                                            onChange={(e) => setIqrLower(parseFloat(e.target.value))}
                                                                        />
                                                                        <Input
                                                                            type="number"
                                                                            className="mt-1 w-1/2 text-xs"
                                                                            value={iqrUpper}
                                                                            onChange={(e) => setIqrUpper(parseFloat(e.target.value))}
                                                                        />
                                                                    </div>
                                                                </div>
                                                            )}

                                                            {outlierMethod === "winsorization" && (
                                                                <div className="mt-2">
                                                                    <span className="text-xs text-white">Winsorization Percentiles:</span>
                                                                    <div className="flex space-x-2">
                                                                        <Input
                                                                            type="number"
                                                                            className="mt-1 w-1/2 text-xs"
                                                                            value={winsorLower}
                                                                            onChange={(e) => setWinsorLower(parseFloat(e.target.value))}
                                                                        />
                                                                        <Input
                                                                            type="number"
                                                                            className="mt-1 w-1/2 text-xs"
                                                                            value={winsorUpper}
                                                                            onChange={(e) => setWinsorUpper(parseFloat(e.target.value))}
                                                                        />
                                                                    </div>
                                                                </div>
                                                            )}
                                                        </>
                                                    ) : (
                                                        <div className="text-center text-white">
                                                            Enable Outlier Removal to select a method.
                                                        </div>
                                                    )}
                                                </>
                                            ) : (
                                                <div className="text-center text-white">
                                                    Please select a train file to choose for outlier detection.
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                </div>
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-1 mt-1">Encoding Technique for Categorical Data</div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">

                                            {trainFile ? (
                                                <>
                                                    <Select onValueChange={setEncodingMethod} value={encodingMethod}>
                                                        <SelectTrigger className="w-full text-xs text-white">
                                                            <SelectValue placeholder="Select Encoding Method" />
                                                        </SelectTrigger>
                                                        <SelectContent>
                                                            <SelectItem value="none">None (Use Only Numeric Columns)</SelectItem>
                                                            <SelectItem value="one-hot">One-Hot Encoding</SelectItem>
                                                            <SelectItem value="label">Label Encoding</SelectItem>
                                                            <SelectItem value="target">Target Encoding</SelectItem>
                                                        </SelectContent>
                                                    </Select>

                                                    {/* Note for users */}
                                                    <div className="text-xs text-gray-400 mt-2 text-center">
                                                        ⚡ If unsure, keep <span className="font-semibold">One-Hot Encoding</span> as default. No effect if no categorical data found.
                                                    </div>
                                                </>
                                            ) : (
                                                <div className="text-center text-white">
                                                    Please select a train file to choose an encoding method.
                                                </div>
                                            )}
                                        </div>
                                    </div>




                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm">Select Data Exploration Techniques</div>
                                            <div className="flex items-center">
                                                <Checkbox
                                                    checked={selectedExplorations.length === availableExplorations.length}
                                                    onCheckedChange={toggleSelectAllExplorations}
                                                />
                                                <span className="ml-2 text-xs">Select All</span>
                                            </div>
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            <div>
                                                {trainFile ? (
                                                    <div className="grid grid-cols-2 gap-1">
                                                        {availableExplorations.map((technique) => (
                                                            <div key={technique} className="flex items-center text-xs">
                                                                <Checkbox
                                                                    checked={selectedExplorations.includes(technique)}
                                                                    onCheckedChange={() => toggleExploration(technique)}
                                                                />
                                                                <span className="ml-1">{technique}</span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    <div className="text-center">Please select a train file to enable data exploration selection.</div>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white"></div>
                                </div>
                            </div>

                        </TabsContent>

                        {/* Terminal Tab Content */}
                        <TabsContent value="terminal">
                            <div
                                ref={terminalRef}
                                className="border border-[rgb(61,68,77)] h-[700px] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 text-sm p-4 overflow-y-auto"
                            >
                                <pre className="whitespace-pre-wrap">{logs || "Terminal Output will be shown here."}</pre>
                            </div>
                        </TabsContent>
                        <TabsContent value="graphs">
                            <div className="flex flex-col items-center justify-center h-[700px] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 text-center p-8 space-y-8">
                                <h2 className="text-5xl font-bold">Graphs will be saved in this Directory</h2>
                                <p className="text-3xl font-semibold">
                                    {datasetPath
                                        ? `${datasetPath.replace(/[/\\]+$/, "")}/linearregression-${trainFile ? trainFile.split(".")[0] : "N/A"}`
                                        : "N/A"}
                                </p>
                            </div>
                        </TabsContent>

                        <TabsContent value="result">
                            <div className="flex flex-col items-center justify-center h-[700px] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 text-center p-4">
                                {results ? (
                                    results
                                        .split("\n")
                                        .filter((line) => line.trim() !== "")
                                        .map((line, index) => (
                                            <div key={index} className="text-5xl font-bold mb-4">
                                                {line}
                                            </div>
                                        ))
                                ) : (
                                    <div className="text-4xl font-bold">Results will be displayed here.</div>
                                )}
                            </div>
                        </TabsContent>


                    </div>
                </Tabs>
            </div>
        </div>
    );
};

export default LinearRegressionComponent;
