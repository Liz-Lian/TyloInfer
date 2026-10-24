


import { create } from 'zustand';


export type ToastType = 'error' | 'success' | 'info' | 'warning';


export interface ToastItemData {

  id: string;

  revision: number;

  type: ToastType;

  message: string;

  title?: string;

  duration?: number;
}


export interface ShowToastOptions {

  id?: string;

  type?: ToastType;

  message: string;

  title?: string;

  duration?: number;
}

interface ToastState {

  toasts: ToastItemData[];






  showToast: (options: ShowToastOptions) => string;






  dismissToast: (id: string, revision?: number) => void;

  clearToasts: () => void;
}

let toastIdCounter = 0;
let toastRevisionCounter = 0;


export const useToastStore = create<ToastState>()((set) => ({
  toasts: [],
  showToast: (options) => {
    const id = options.id ?? `toast-${Date.now()}-${++toastIdCounter}`;
    const defaultDuration = options.type === 'error' ? 5000 : 3500;
    const duration = options.duration !== undefined ? options.duration : defaultDuration;
    const item: ToastItemData = {
      id,
      revision: ++toastRevisionCounter,
      type: options.type ?? 'info',
      message: options.message,
      title: options.title,
      duration,
    };
    set((state) => ({
      toasts: [...state.toasts.filter((t) => t.id !== id), item],
    }));
    return id;
  },
  dismissToast: (id, revision) => {
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id || (revision !== undefined && t.revision !== revision)),
    }));
  },
  clearToasts: () => set({ toasts: [] }),
}));




export const toast = {






  error: (message: string, options?: { title?: string; duration?: number; id?: string }) =>
    useToastStore.getState().showToast({ ...options, type: 'error', message }),







  success: (message: string, options?: { title?: string; duration?: number; id?: string }) =>
    useToastStore.getState().showToast({ ...options, type: 'success', message }),







  info: (message: string, options?: { title?: string; duration?: number; id?: string }) =>
    useToastStore.getState().showToast({ ...options, type: 'info', message }),







  warning: (message: string, options?: { title?: string; duration?: number; id?: string }) =>
    useToastStore.getState().showToast({ ...options, type: 'warning', message }),






  show: (options: ShowToastOptions) => useToastStore.getState().showToast(options),






  dismiss: (id: string) => useToastStore.getState().dismissToast(id),


  clear: () => useToastStore.getState().clearToasts(),
};
