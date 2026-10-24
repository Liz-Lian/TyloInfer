
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { XIcon } from '@phosphor-icons/react';
import { useRef, type ReactNode, type RefObject } from 'react';

interface DialogProps {
  open: boolean;

  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  closeLabel: string;
  className?: string;
  children: ReactNode;

  returnFocusRef?: RefObject<HTMLElement | null>;
}


export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  closeLabel,
  className = '',
  children,
  returnFocusRef,
}: DialogProps) {
  const previousFocus = useRef<HTMLElement | null>(null);

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="dialog-backdrop">
          <DialogPrimitive.Content
            className={`studio-dialog ${className}`.trim()}
            {...(!description ? { 'aria-describedby': undefined } : {})}
            onOpenAutoFocus={() => {
              previousFocus.current =
                document.activeElement instanceof HTMLElement ? document.activeElement : null;
            }}
            onCloseAutoFocus={(event) => {
              const target = returnFocusRef?.current ?? previousFocus.current;
              if (target?.isConnected) {
                event.preventDefault();
                target.focus({ preventScroll: true });
              }
            }}
          >
            <header>
              <div>
                <DialogPrimitive.Title asChild>
                  <h2>{title}</h2>
                </DialogPrimitive.Title>
                {description && (
                  <DialogPrimitive.Description asChild>
                    <p>{description}</p>
                  </DialogPrimitive.Description>
                )}
              </div>
              <DialogPrimitive.Close asChild>
                <button type="button" aria-label={closeLabel}>
                  <XIcon size={18} aria-hidden="true" />
                </button>
              </DialogPrimitive.Close>
            </header>
            {children}
          </DialogPrimitive.Content>
        </DialogPrimitive.Overlay>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
