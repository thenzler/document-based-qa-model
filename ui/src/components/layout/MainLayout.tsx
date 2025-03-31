import React from 'react';
import { AppBar, Drawer, List, ListItem, Toolbar, Typography, IconButton, useMediaQuery, useTheme } from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import MenuIcon from '@mui/icons-material/Menu';
import DashboardIcon from '@mui/icons-material/Dashboard';
import DescriptionIcon from '@mui/icons-material/Description';
import SchoolIcon from '@mui/icons-material/School';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import AssessmentIcon from '@mui/icons-material/Assessment';

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [drawerOpen, setDrawerOpen] = React.useState(false);

  const menuItems = [
    { title: 'Dashboard', path: '/', icon: <DashboardIcon /> },
    { title: 'Dokumente', path: '/documents', icon: <DescriptionIcon /> },
    { title: 'Training', path: '/training', icon: <SchoolIcon /> },
    { title: 'Modelle', path: '/models', icon: <ModelTrainingIcon /> },
    { title: 'Churn Prediction', path: '/churn', icon: <AssessmentIcon /> }
  ];

  const toggleDrawer = (open: boolean) => (event: React.KeyboardEvent | React.MouseEvent) => {
    if (
      event.type === 'keydown' &&
      ((event as React.KeyboardEvent).key === 'Tab' || (event as React.KeyboardEvent).key === 'Shift')
    ) {
      return;
    }
    setDrawerOpen(open);
  };

  const handleNavigation = (path: string) => {
    navigate(path);
    if (isMobile) {
      setDrawerOpen(false);
    }
  };

  return (
    <div className="app-container">
      <AppBar position="fixed">
        <Toolbar>
          {isMobile && (
            <IconButton
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={toggleDrawer(true)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Dokumentenbasiertes Frage-Antwort-System
          </Typography>
          {!isMobile && (
            <div className="nav-tabs">
              {menuItems.map((item) => (
                <div
                  key={item.path}
                  className={`nav-tab ${location.pathname === item.path ? 'active' : ''}`}
                  onClick={() => handleNavigation(item.path)}
                >
                  {item.title}
                </div>
              ))}
            </div>
          )}
        </Toolbar>
      </AppBar>

      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={toggleDrawer(false)}
      >
        <List sx={{ width: 250 }}>
          {menuItems.map((item) => (
            <ListItem
              button
              key={item.path}
              onClick={() => handleNavigation(item.path)}
              selected={location.pathname === item.path}
            >
              {item.icon}
              <Typography sx={{ ml: 2 }}>{item.title}</Typography>
            </ListItem>
          ))}
        </List>
      </Drawer>

      <main className="main-content">
        {children}
      </main>
    </div>
  );
};

export default MainLayout;
