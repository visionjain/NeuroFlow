"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import axios from "axios";
import { useParams } from "next/navigation";
import Nav from "@/components/navbar/page";
import CopyRight from "@/components/copybar/page";
import Loader from "@/components/loader/page";
import LinearRegressionComponent from "@/components/AlgoComps/LinearRegression.tsx/page";
import LogisticRegressionComponent from "@/components/AlgoComps/LogisticRegression.tsx/page";


const ProjectPage = () => {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [userDetails, setUserDetails] = useState<any>(null);
  const [userRole, setUserRole] = useState("");
  const { id } = useParams();
  const [projectDetails, setProjectDetails] = useState<any>(null);

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
            ProjectAlgo: project.algorithm,
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
          {projectDetails.ProjectAlgo === "Linear regression" && (
            <LinearRegressionComponent
              projectName={projectDetails.projectName}
              projectAlgo={projectDetails.ProjectAlgo}
              projectTime={projectDetails.ProjectTime}
              projectId={projectDetails.projectId}
            />
          )}
          {projectDetails.ProjectAlgo === "Logistic regression" && (
            <LogisticRegressionComponent
              projectName={projectDetails.projectName}
              projectAlgo={projectDetails.ProjectAlgo}
              projectTime={projectDetails.ProjectTime}
              projectId={projectDetails.projectId}
            />
          )}
          {projectDetails.ProjectAlgo === "knn" && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <h2 className="text-3xl font-bold mb-4">🚧 KNN Algorithm</h2>
                <p className="text-xl text-gray-500">Coming Soon...</p>
              </div>
            </div>
          )}
        </div>
        <CopyRight />
      </div>
    </div>
  );
};

export default ProjectPage;
