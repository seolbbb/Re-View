import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { AuthProvider } from './context/AuthContext';
import { VideoProvider } from './context/VideoContext';
import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import HomePage from './pages/HomePage';
import AnalysisPage from './pages/AnalysisPage';
import LoadingPage from './pages/LoadingPage';
import './index.css';

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <VideoProvider>
          <BrowserRouter>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route path="/signup" element={<SignupPage />} />
              <Route path="/" element={
                <ProtectedRoute><HomePage /></ProtectedRoute>
              } />
              <Route path="/analysis/:id" element={
                <ProtectedRoute><AnalysisPage /></ProtectedRoute>
              } />
              <Route path="/loading" element={
                <ProtectedRoute><LoadingPage /></ProtectedRoute>
              } />
            </Routes>
          </BrowserRouter>
        </VideoProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
