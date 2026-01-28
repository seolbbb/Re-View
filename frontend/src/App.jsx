import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { VideoProvider } from './context/VideoContext';
import LoginPage from './pages/LoginPage';
import HomePage from './pages/HomePage';
import AnalysisPage from './pages/AnalysisPage';
import LoadingPage from './pages/LoadingPage';
import './index.css';

function App() {
  return (
    <ThemeProvider>
      <VideoProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/" element={<HomePage />} />
            <Route path="/analysis/:id" element={<AnalysisPage />} />
            <Route path="/loading" element={<LoadingPage />} />
          </Routes>
        </BrowserRouter>
      </VideoProvider>
    </ThemeProvider>
  );
}

export default App;
