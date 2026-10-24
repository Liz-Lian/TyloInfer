
import * as SwitchPrimitive from '@radix-ui/react-switch';
import './primitives.css';

interface SwitchProps {
  checked: boolean;

  onCheckedChange: (checked: boolean) => void;
  label: string;
  disabled?: boolean;
  id?: string;
}


export function Switch({ checked, onCheckedChange, label, disabled, id }: SwitchProps) {
  return (
    <SwitchPrimitive.Root
      id={id}
      className="studio-switch"
      checked={checked}
      onCheckedChange={onCheckedChange}
      disabled={disabled}
      aria-label={label}
      type="button"
    >
      <SwitchPrimitive.Thumb className="studio-switch-thumb" />
    </SwitchPrimitive.Root>
  );
}
