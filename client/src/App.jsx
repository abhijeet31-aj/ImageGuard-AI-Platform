import { useState } from "react";
import Dashboard from "./pages/Dashboard";
import Login from "./pages/Login";
import Result from "./pages/Result";
import Upload from "./pages/Upload";
import History from "./pages/History";
import Register from "./pages/Register";


function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(() =>
    Boolean(localStorage.getItem("token"))
  );
  const [view, setView] = useState("dashboard");
  const [latestAnalysis, setLatestAnalysis] = useState(null);
  const [latestPreview, setLatestPreview] = useState("");

  const [authScreen, setAuthScreen] = useState("login");

  const handleLogout = () => {
    localStorage.removeItem("token");
    setIsLoggedIn(false);
    setView("dashboard");
    setLatestAnalysis(null);
    setLatestPreview("");
  };

  const handleUploadComplete = (analysis, preview) => {
    setLatestAnalysis(analysis);
    setLatestPreview(preview);
    setView("result");
  };

  if (!isLoggedIn) {
    return authScreen === "register" ? (
      <Register onGoLogin={() => setAuthScreen("login")} />
    ) : (
      <Login
        onLogin={() => setIsLoggedIn(true)}
        onRegister={() => setAuthScreen("register")}
      />
    );
  }

  if (view === "upload") {
    return (
      <Upload
        onBack={() => setView("dashboard")}
        onUploadComplete={handleUploadComplete}
      />
    );
  }

  if (view === "result") {
    return (
      <Result
        analysis={latestAnalysis}
        imagePreview={latestPreview}
        onBack={() => setView("dashboard")}
        onAnalyzeAgain={() => setView("upload")}
      />
    );
  }

  if (view === "history") {
    return (
      <History
        onBack={() => setView("dashboard")}
        onViewResult={(analysis, imageUrl) => {
          setLatestAnalysis(analysis);
          setLatestPreview(imageUrl);
          setView("result");
        }}
      />
    );
  }

  return <Dashboard onLogout={handleLogout} onNavigate={setView} />;
}

export default App;