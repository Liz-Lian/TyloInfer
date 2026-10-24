


import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import type { ReactElement, ReactNode } from 'react';
import './primitives.css';


export interface TooltipProps {

  content: ReactNode;

  children: ReactElement;
}


export const TooltipProvider = TooltipPrimitive.Provider;






export function Tooltip({ content, children }: TooltipProps) {
  return (
    <TooltipPrimitive.Root>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content className="studio-tooltip" sideOffset={6} collisionPadding={12}>
          {content}
          <TooltipPrimitive.Arrow className="studio-tooltip-arrow" />
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  );
}
