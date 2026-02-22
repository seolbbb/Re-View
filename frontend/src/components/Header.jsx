import { Link, useNavigate } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { useAuth } from '../context/AuthContext';
import { Sun, Moon, LogOut } from 'lucide-react';
import './Header.css';

function Header() {
  const navigate = useNavigate();
  const { theme, toggleTheme } = useTheme();
  const { user, signOut } = useAuth();

  const handleLogout = async () => {
    try {
      await signOut();
      navigate('/login');
    } catch (err) {
      console.error('로그아웃 실패:', err);
    }
  };

  return (
    <header className="header">
      <div className="header-container">
        <Link to="/" className="logo">
          <span className="logo-re">Re:</span>
          <span className="logo-view">View</span>
          <span className="logo-play">▶</span>
        </Link>



        <div className="header-actions">
          {user && (
            <span className="user-email">{user.email}</span>
          )}
          <button className="theme-toggle" onClick={toggleTheme} aria-label="테마 변경">
            {theme === 'dark' ? (
              <Sun className="w-5 h-5" />
            ) : (
              <Moon className="w-5 h-5" />
            )}
          </button>
          <button className="logout-btn" onClick={handleLogout} title="로그아웃">
            <LogOut className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  );
}

export default Header;
