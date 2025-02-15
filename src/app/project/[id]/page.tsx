"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import axios from "axios";
import { useParams } from "next/navigation";
import Nav from "@/components/navbar/page";
import CopyRight from "@/components/copybar/page";
import Loader from "@/components/loader/page";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";

const ProjectPage = () => {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [userDetails, setUserDetails] = useState<any>(null);
  const [userRole, setUserRole] = useState("");
  const { id } = useParams();
  const [projectDetails, setProjectDetails] = useState<any>(null);
  const [trainFile, setTrainFile] = useState<string | null>(null);
  const [testFile, setTestFile] = useState<string | null>(null);
  const [showTestUpload, setShowTestUpload] = useState(true); // Toggle for test dataset button

  const trainInputRef = useRef<HTMLInputElement>(null);
  const testInputRef = useRef<HTMLInputElement>(null);

  const formatProjectTime = (isoString: string) => {
    const date = new Date(isoString);
    return `${date.toLocaleDateString("en-IN")} ${date.toLocaleTimeString("en-IN", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    })}`;
  };

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const response = await axios.get("/api/users/me");
        const { data } = response.data;
        setUserDetails(data);
        setUserRole(data.role);
        setLoading(false);

        const project = data.projects.find((project: any) => project._id.toString() === id);
        if (project) {
          setProjectDetails({
            projectId: project._id,
            projectName: project.topic,
            ProjectTime: formatProjectTime(project.createdAt),
          });
        } else {
          console.error("Project not found");
        }
      } catch (error) {
        expirylogout();
        router.push("/login");
      }
    };

    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/login");
    } else {
      fetchUserData();
    }
  }, [router, id]);

  const expirylogout = async () => {
    try {
      await axios.get("/api/users/logout");
      localStorage.removeItem("token");
    } catch (error) {
      console.error("Error logging out:", error);
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
      // Clear both files when toggling
      setTrainFile(null);
      setTestFile(null);

      // Reset both file inputs
      if (trainInputRef.current) trainInputRef.current.value = "";
      if (testInputRef.current) testInputRef.current.value = "";

      return !prev;
    });
  };

  if (!projectDetails) {
    return (
      <div className="flex h-screen justify-center items-center">
        <Loader />
      </div>
    );
  }

  return (
    <div>
      <div className="h-[100vh] pt-24">
        <Nav loading={loading} userRole={userRole} userDetails={userDetails} />
        <div className="h-[90%] dark:bg-[#212628] rounded-3xl ml-8 bg-white mr-8 overflow-y-auto" style={{ maxHeight: "90vh" }}>
          <div>
            <div className="text-xl">
              <div className="pl-4 pt-4 font-bold">
                <h1 className="italic text-3xl">
                  {projectDetails.projectName}{" "}
                  <span className="text-base lowercase">{projectDetails.ProjectTime}</span>
                </h1>
              </div>

              {/* Buttons Section */}
              <div className="border border-[rgb(61,68,77)] flex flex-col gap-6 dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 p-4">
                <div className="flex">
                  <div className="dark:bg-[#212628] rounded-xl w-[25%] bg-white">
                    <div className="flex justify-center items-center gap-8 mt-4 mb-3">
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
                              <p className="text-sm text-gray-500 dark:text-grey-600 dark:group-hover:text-white mt-2">{trainFile}</p>
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
                                <p className="text-sm text-gray-500 dark:text-grey-600 dark:group-hover:text-white mt-2">{testFile}</p>
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

                    {/* Toggle Button for Test Dataset */}
                    <p className="underline mb-3 flex justify-center text-sm text-blue-600 cursor-pointer" onClick={toggleTestDataset}>
                      {showTestUpload ? "Don't have a test dataset?" : "Have a test dataset?"}
                    </p>
                  </div>
                  <div className="dark:bg-[#212628] rounded-xl w-[25%] bg-white ml-4">
                    
                  </div>
                </div>
              </div>
              <div className="ml-4 text-xl font-bold italic mt-2">TERMINAL</div>
              <div className="border border-[rgb(61,68,77)] h-60 flex flex-col gap-6 dark:bg-[#0E0E0E] bg-[#E6E6E6] rounded-xl ml-4 mr-4 text-sm p-4 overflow-y-auto">

                Terminal Output will be here.


              </div>
            </div>
          </div>
        </div>
        <CopyRight />
      </div>
    </div>
  );
};

export default ProjectPage;
