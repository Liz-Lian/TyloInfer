



import { create } from 'zustand';

interface UiState {
  sidebarOpen: boolean;

  closeSidebar: () => void;

  openSidebar: () => void;

  toggleSidebar: () => void;
}


export const useUiStore = create<UiState>()((set) => ({
  sidebarOpen: false,
  closeSidebar: () => set({ sidebarOpen: false }),
  openSidebar: () => set({ sidebarOpen: true }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
}));
