"use client";

import React, { useState, useEffect, useRef } from "react";
import { FaPlay, FaSpinner } from "react-icons/fa";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { useRouter } from "next/navigation";

interface KNNProps {
    projectName: string;
    projectAlgo: string;
    projectTime: string;
    projectId: string;
}

const KNNComponent: React.FC<KNNProps> = ({ projectName, projectAlgo, projectTime, projectId }) => {
    const router = useRouter();
    
    // Dataset & File Management
    const [trainFile, setTrainFile] = useState<string | null>(null);
    const [testFile, setTestFile] = useState<string | null>(null);
    const [datasetPath, setDatasetPath] = useState<string>("");
    const [showTestUpload, setShowTestUpload] = useState(true);
    const [testSplitRatio, setTestSplitRatio] = useState<string>("0.2");
    
    // Column Selection
    const [trainColumns, setTrainColumns] = useState<string[]>([]);
    const [selectedTrainColumns, setSelectedTrainColumns] = useState<string[]>([]);
    const [selectedOutputColumn, setSelectedOutputColumn] = useState<string | null>(null);
    
    // KNN Configuration
    const [taskType, setTaskType] = useState<string>("classification");
    const [kValue, setKValue] = useState<string>("5");
    const [enableAutoK, setEnableAutoK] = useState(false);
    const [kRangeStart, setKRangeStart] = useState<string>("3");
    const [kRangeEnd, setKRangeEnd] = useState<string>("15");
    const [distanceMetric, setDistanceMetric] = useState<string>("euclidean");
    const [weights, setWeights] = useState<string>("uniform");
    const [algorithm, setAlgorithm] = useState<string>("auto");
    const [leafSize, setLeafSize] = useState<string>("30");
    const [pValue, setPValue] = useState<string>("2");
    
    // Preprocessing
    const [selectedHandlingMissingValue, setSelectedHandlingMissingValue] = useState<string>("drop");
    const [removeDuplicates, setRemoveDuplicates] = useState(true);
    const [encodingMethod, setEncodingMethod] = useState("onehot");
    const [selectedFeatureScaling, setSelectedFeatureScaling] = useState<string>("standard");
    
    // Outlier Detection
    const [enableOutlierDetection, setEnableOutlierDetection] = useState(false);
    const [outlierMethod, setOutlierMethod] = useState("");
    const [zScoreThreshold, setZScoreThreshold] = useState(3.0);
    
    // Advanced Options
    const [enableCV, setEnableCV] = useState(false);
    const [cvFolds, setCvFolds] = useState("5");
    const [enableDimReduction, setEnableDimReduction] = useState(false);
    const [dimReductionMethod, setDimReductionMethod] = useState("pca");
    const [nComponents, setNComponents] = useState("2");
    const [enableImbalance, setEnableImbalance] = useState(false);
    
    // Data Exploration
    const [selectedExplorations, setSelectedExplorations] = useState<string[]>([]);
    
    // Graph Selection
    const [selectedGraphs, setSelectedGraphs] = useState<string[]>([]);
    
    // UI State
    const [isRunning, setIsRunning] = useState<boolean>(false);
    const [logs, setLogs] = useState<string>("");
    const [results, setResults] = useState<string>("");
    const [generatedGraphs, setGeneratedGraphs] = useState<string[]>([]);
    const [modelTrained, setModelTrained] = useState<boolean>(false);
    const [missingFilesWarning, setMissingFilesWarning] = useState<string[]>([]);
    
    const terminalRef = useRef<HTMLDivElement>(null);
    
    const availableExplorations = [
        "First 5 Rows",
        "Last 5 Rows",
        "Dataset Shape",
        "Data Types",
        "Summary Statistics",
        "Missing Values",
        "Unique Values Per Column",
        "Duplicate Rows",
        "Correlation Matrix",
        "Target Distribution"
    ];
    
    const handleRunScript = () => {
        toast.info("KNN training will be implemented in next phase", {
            style: { background: 'blue', color: 'white' }
        });
    };

    return (
        <div>
            <div className="text-xl">
                <Tabs defaultValue="home">
                    <div className="flex items-center justify-between px-4 mt-2">
                        <div className="font-bold flex items-center gap-3">
                            <Button
                                onClick={() => router.push('/')}
                                className="rounded-xl border-2 border-[rgb(61,68,77)] bg-white dark:bg-[#0E0E0E] hover:bg-gray-100 dark:hover:bg-[#1a1a1a] text-black dark:text-white shadow-md"
                                title="Back to Home"
                            >
                                ← Back
                            </Button>
                            <h1 className="italic text-2xl">
                                {projectName} - {projectAlgo}{" "}
                                <span className="text-sm lowercase">{projectTime}</span>
                            </h1>
                        </div>

                        <TabsList className="flex w-[50%] text-black dark:text-white bg-[#e6e6e6] dark:bg-[#0F0F0F]">
                            <TabsTrigger className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]" value="home">
                                Home
                            </TabsTrigger>
                            <TabsTrigger className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]" value="graphs">
                                Graphs
                            </TabsTrigger>
                            <TabsTrigger className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]" value="result">
                                Results
                            </TabsTrigger>
                            <TabsTrigger className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628]" value="terminal">
                                Terminal
                            </TabsTrigger>
                            <TabsTrigger className="w-[20%] border border-transparent data-[state=active]:border-[rgb(61,68,77)] data-[state=active]:rounded-md data-[state=active]:bg-[#212628] disabled:opacity-50 disabled:cursor-not-allowed" value="predict" disabled={!modelTrained}>
                                {modelTrained ? "🔮 Predict" : "🔒 Predict"}
                            </TabsTrigger>
                        </TabsList>

                        <div className="flex gap-2">
                            <Button className="rounded-xl" onClick={handleRunScript} disabled={isRunning}>
                                {isRunning ? <FaSpinner className="animate-spin" /> : <FaPlay />}
                            </Button>
                        </div>
                    </div>

                    <div className="mt-2">
                        <TabsContent value="home">
                            <div className="border border-[rgb(61,68,77)] flex flex-col gap-3 dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-4">
                                
                                {/* Dataset Compatibility Info */}
                                <div className="dark:bg-[#1a1d1f] bg-[#f5f5f5] rounded-xl p-4 border-2 border-blue-500 dark:border-blue-600">
                                    <h3 className="text-lg font-bold mb-2 text-center">📋 K-Nearest Neighbors - Supported Dataset Types</h3>
                                    <div className="grid grid-cols-3 gap-4 text-sm">
                                        <div>
                                            <p className="font-semibold text-blue-600 dark:text-blue-400">✅ Classification:</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Binary/Multi-class categories (Iris, Wine, Digits)</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Examples: Disease diagnosis, Customer segmentation</p>
                                        </div>
                                        <div>
                                            <p className="font-semibold text-green-600 dark:text-green-400">✅ Regression:</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Continuous numeric prediction (House prices, Stock)</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Examples: Real estate, Sales forecast</p>
                                        </div>
                                        <div>
                                            <p className="font-semibold text-red-600 dark:text-red-400">⚠️ Important:</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Feature Scaling: MANDATORY (distance-based)</p>
                                            <p className="text-xs text-gray-600 dark:text-gray-400">Works best with &lt;20 features</p>
                                        </div>
                                    </div>
                                </div>

                                {/* First Row - Dataset Upload */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-4">
                                        <div className="mb-4 text-center">
                                            <Label className="text-sm font-semibold">Dataset Directory Path</Label>
                                            <Input type="text" placeholder="Ex: D:\datasetpath" className="mt-1 dark:bg-[#0F0F0F]" />
                                        </div>
                                        <div className="flex w-full gap-2">
                                            <div className="flex flex-col w-full items-center">
                                                <Label className="text-sm font-semibold mb-1">Train Data</Label>
                                                <Button className="h-12 w-full border-2 border-dashed border-gray-500 rounded-md">
                                                    <span className="text-3xl">+</span>
                                                </Button>
                                            </div>
                                            <div className="flex flex-col w-full items-center">
                                                <Label className="text-sm font-semibold mb-1">Test Data</Label>
                                                <Button className="h-12 w-full border-2 border-dashed border-gray-500 rounded-md">
                                                    <span className="text-3xl">+</span>
                                                </Button>
                                            </div>
                                        </div>
                                        <p className="underline mt-2 flex justify-center text-sm text-blue-600 cursor-pointer">
                                            Don't have test dataset?
                                        </p>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm">Select Train Columns</div>
                                            <div className="flex items-center">
                                                <Checkbox />
                                                <span className="ml-1 text-xs">Select All</span>
                                            </div>
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl text-center text-sm text-gray-500">
                                            Select train file to enable column selection
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-1 mt-1">Select Output Column</div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl text-center text-sm text-gray-500">
                                            Select train file to enable output selection
                                        </div>
                                    </div>
                                </div>

                                {/* Second Row - Data Exploration & Preprocessing */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm">Data Exploration</div>
                                            <div className="flex items-center">
                                                <Checkbox />
                                                <span className="ml-1 text-xs">Select All</span>
                                            </div>
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            <div className="space-y-1 text-xs">
                                                <div className="flex items-center"><Checkbox disabled /><span className="ml-1">First 5 Rows</span></div>
                                                <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Dataset Shape</span></div>
                                                <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Summary Statistics</span></div>
                                                <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Missing Values</span></div>
                                                <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Correlation Matrix</span></div>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-2 mt-1">Data Cleaning</div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl space-y-3">
                                            <div>
                                                <Label className="text-xs font-semibold">Handle Missing Values</Label>
                                                <Select>
                                                    <SelectTrigger className="h-8 text-xs dark:bg-[#212628]">
                                                        <SelectValue placeholder="Drop Rows" />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="drop">Drop Rows with Missing Values</SelectItem>
                                                        <SelectItem value="mean">Mean Imputation</SelectItem>
                                                        <SelectItem value="median">Median Imputation</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <Checkbox id="remove-dup" />
                                                <Label htmlFor="remove-dup" className="text-xs">Remove Duplicates</Label>
                                            </div>
                                            <div>
                                                <Label className="text-xs font-semibold">Categorical Encoding</Label>
                                                <Select>
                                                    <SelectTrigger className="h-8 text-xs dark:bg-[#212628]">
                                                        <SelectValue placeholder="One-Hot" />
                                                    </SelectTrigger>
                                                </Select>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-2 mt-1">Outlier Detection</div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl space-y-2">
                                            <div className="flex items-center space-x-2">
                                                <Checkbox id="enable-outlier" />
                                                <Label htmlFor="enable-outlier" className="text-xs">Enable Outlier Detection</Label>
                                            </div>
                                            <div>
                                                <Label className="text-xs font-semibold">Detection Method</Label>
                                                <Select disabled>
                                                    <SelectTrigger className="h-8 text-xs dark:bg-[#212628]">
                                                        <SelectValue placeholder="Select method" />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="zscore">Z-Score</SelectItem>
                                                        <SelectItem value="iqr">IQR</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Third Row - Feature Scaling & KNN Config */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-2 mt-1 flex items-center">
                                            <span className="text-red-500 mr-1">*</span>
                                            Feature Scaling (REQUIRED)
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl space-y-2">
                                            <div className="bg-red-50 dark:bg-red-900/20 border border-red-300 rounded p-2 text-xs text-center">
                                                ⚠️ KNN requires feature scaling
                                            </div>
                                            <div className="space-y-1 text-xs">
                                                <div className="flex items-center"><Checkbox /><span className="ml-1">Min-Max Scaling</span></div>
                                                <div className="flex items-center"><Checkbox checked /><span className="ml-1">Standard Scaling (Recommended)</span></div>
                                                <div className="flex items-center"><Checkbox /><span className="ml-1">Robust Scaling</span></div>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-2 mt-1">KNN Algorithm Config</div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl space-y-2 overflow-auto">
                                            <div>
                                                <Label className="text-xs font-semibold">K Value (Neighbors)</Label>
                                                <Input type="number" placeholder="5" className="h-8 text-xs dark:bg-[#212628]" />
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <Checkbox id="auto-k" />
                                                <Label htmlFor="auto-k" className="text-xs">Auto-Find Optimal K</Label>
                                            </div>
                                            <div>
                                                <Label className="text-xs font-semibold">Distance Metric</Label>
                                                <Select>
                                                    <SelectTrigger className="h-8 text-xs dark:bg-[#212628]">
                                                        <SelectValue placeholder="Euclidean" />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="euclidean">Euclidean</SelectItem>
                                                        <SelectItem value="manhattan">Manhattan</SelectItem>
                                                        <SelectItem value="minkowski">Minkowski</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            <div>
                                                <Label className="text-xs font-semibold">Weight Function</Label>
                                                <Select>
                                                    <SelectTrigger className="h-8 text-xs dark:bg-[#212628]">
                                                        <SelectValue placeholder="Uniform" />
                                                    </SelectTrigger>
                                                </Select>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-1/3 bg-white p-2">
                                        <div className="font-semibold text-sm mb-2 mt-1">Advanced Options</div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl space-y-2 overflow-auto">
                                            <div>
                                                <Label className="text-xs font-semibold">Algorithm Type</Label>
                                                <Select>
                                                    <SelectTrigger className="h-8 text-xs dark:bg-[#212628]">
                                                        <SelectValue placeholder="Auto" />
                                                    </SelectTrigger>
                                                    <SelectContent>
                                                        <SelectItem value="auto">Auto</SelectItem>
                                                        <SelectItem value="ball_tree">Ball Tree</SelectItem>
                                                        <SelectItem value="kd_tree">KD Tree</SelectItem>
                                                    </SelectContent>
                                                </Select>
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <Checkbox id="enable-cv" />
                                                <Label htmlFor="enable-cv" className="text-xs">Enable Cross-Validation</Label>
                                            </div>
                                            <div>
                                                <Label className="text-xs font-semibold">CV Folds</Label>
                                                <Input type="number" placeholder="5" className="h-8 text-xs dark:bg-[#212628]" disabled />
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <Checkbox id="dim-reduction" />
                                                <Label htmlFor="dim-reduction" className="text-xs">Dimensionality Reduction (PCA)</Label>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Fourth Row - Graph Selection */}
                                <div className="flex gap-x-3">
                                    <div className="dark:bg-[#212628] h-52 rounded-xl w-full bg-white p-2">
                                        <div className="flex items-center justify-between mb-1 mt-1">
                                            <div className="font-semibold text-sm">Select Graphs to Generate (20+ Available)</div>
                                            <div className="flex items-center">
                                                <Checkbox />
                                                <span className="ml-1 text-xs">Select All</span>
                                            </div>
                                        </div>
                                        <div className="dark:bg-[#0E0E0E] bg-[#E6E6E6] h-40 p-3 rounded-xl overflow-auto">
                                            <div className="grid grid-cols-4 gap-2 text-xs">
                                                <div className="space-y-1">
                                                    <div className="font-semibold text-blue-600 dark:text-blue-400">KNN Specific</div>
                                                    <div className="flex items-center"><Checkbox disabled /><span className="ml-1">K vs Accuracy</span></div>
                                                    <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Distance Distribution</span></div>
                                                    <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Decision Boundary</span></div>
                                                    <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Neighbor Analysis</span></div>
                                                </div>
                                                <div className="space-y-1">
                                                    <div className="font-semibold text-green-600 dark:text-green-400">Classification</div>
                                                    <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Confusion Matrix</span></div>
                                                    <div className="flex items-center"><Checkbox disabled /><span className="ml-1">ROC Curve</span></div>
                                                    <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Precision-Recall</span></div>
                                                </div>
                                                <div className="space-y-1">
                                                    <div className="font-semibold text-purple-600 dark:text-purple-400">Regression</div>
                                                    <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Residual Plot</span></div>
                                                    <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Predicted vs Actual</span></div>
                                                    <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Error Distribution</span></div>
                                                </div>
                                                <div className="space-y-1">
                                                    <div className="font-semibold text-orange-600 dark:text-orange-400">Feature Analysis</div>
                                                    <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Correlation Heatmap</span></div>
                                                    <div className="flex items-center"><Checkbox disabled /><span className="ml-1">PCA Visualization</span></div>
                                                    <div className="flex items-center"><Checkbox disabled /><span className="ml-1">Box Plots</span></div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                            </div>
                        </TabsContent>

                        <TabsContent value="terminal">
                            <div className="ml-4 mr-4">
                                <div ref={terminalRef} className="border border-[rgb(61,68,77)] h-[640px] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl text-sm p-4 overflow-y-auto">
                                    <pre className="whitespace-pre-wrap">{logs || "Terminal Output will be shown here."}</pre>
                                </div>
                            </div>
                        </TabsContent>

                        <TabsContent value="result">
                            <div className="ml-4 mr-4 min-h-[640px] border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl p-6">
                                <h2 className="text-2xl font-bold mb-4">📊 Model Results</h2>
                                <p className="text-gray-500 dark:text-gray-400">Results will appear here after training</p>
                            </div>
                        </TabsContent>

                        <TabsContent value="graphs">
                            <div className="ml-4 mr-4 min-h-[640px] border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl p-6">
                                <h2 className="text-2xl font-bold mb-4">📈 Visualizations</h2>
                                <p className="text-gray-500 dark:text-gray-400">Graphs will appear here after training</p>
                            </div>
                        </TabsContent>

                        <TabsContent value="predict">
                            <div className="ml-4 mr-4 min-h-[640px] border border-[rgb(61,68,77)] dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl p-6">
                                <h2 className="text-2xl font-bold mb-4">🔮 Make Predictions</h2>
                                <p className="text-gray-500 dark:text-gray-400">Prediction interface will appear here</p>
                            </div>
                        </TabsContent>
                    </div>
                </Tabs>
            </div>
        </div>
    );
};

export default KNNComponent;
