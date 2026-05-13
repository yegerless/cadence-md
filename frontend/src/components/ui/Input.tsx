import type { InputHTMLAttributes } from 'react'

type InputProps = InputHTMLAttributes<HTMLInputElement> & {
  label: string
  hint?: string
}

export function Input({ label, hint, id, className = '', ...props }: InputProps) {
  const inputId = id ?? props.name ?? label

  return (
    <label className={`field ${className}`.trim()} htmlFor={inputId}>
      <span>{label}</span>
      <input id={inputId} {...props} />
      {hint ? <small>{hint}</small> : null}
    </label>
  )
}
