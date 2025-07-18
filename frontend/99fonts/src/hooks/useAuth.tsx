import { useState, useEffect } from 'react';
import { url_base } from '../utils';

interface AuthState {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: any;
  token: string | null;
}

export const useAuth = () => {
  const [authState, setAuthState] = useState<AuthState>({
    isAuthenticated: false,
    isLoading: true,
    user: null,
    token: null
  });

  const checkAuthStatus = async () => {
    const token = localStorage.getItem('authToken');
    console.debug("token: ", token);

    if (!token) {
      console.debug("No token found, setting unauthenticated");
      setAuthState({
        isAuthenticated: false,
        isLoading: false,
        user: null,
        token: null
      });
      return;
    }

    try {
      console.debug("Checking auth status with token:", token);
      const response = await fetch(`${url_base}/api/auth/verify-auth-token`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const userData = await response.json();
        setAuthState({
          isAuthenticated: true,
          isLoading: false,
          user: userData,
          token: token
        });
        console.debug("Auth state updated: authenticated");
      } else {
        // Token is invalid
        console.debug("Token is invalid, response status:", response.status);
        localStorage.removeItem('authToken');
        localStorage.removeItem('userId');
        setAuthState({
          isAuthenticated: false,
          isLoading: false,
          user: null,
          token: null
        });
      }
    } catch (error) {
      // Network error, clear token
      console.debug("Network error during auth check:", error);
      localStorage.removeItem('authToken');
      localStorage.removeItem('userId');
      setAuthState({
        isAuthenticated: false,
        isLoading: false,
        user: null,
        token: null
      });
    }
  };

  const login = (token: string, userId: string) => {
    console.debug("Login called with token:", token, "userId:", userId);
    localStorage.setItem('authToken', token);
    localStorage.setItem('userId', userId);
    setAuthState(prev => ({
      ...prev,
      isAuthenticated: true,
      token: token
    }));
    console.debug("Auth state updated after login");
  };

  const logout = () => {
    console.debug("Logout called");
    localStorage.removeItem('authToken');
    localStorage.removeItem('userId');
    setAuthState({
      isAuthenticated: false,
      isLoading: false,
      user: null,
      token: null
    });
  };

  useEffect(() => {
    console.debug("useAuth effect running, checking auth status");
    checkAuthStatus();
  }, []);

  useEffect(() => {
    console.debug("Auth state changed:", authState);
  }, [authState]);

  return {
    ...authState,
    login,
    logout,
    checkAuthStatus
  };
}; 