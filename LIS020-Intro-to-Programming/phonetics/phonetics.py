def hz2bark (f: int|float) -> float:
  b = (26.81 * f)/(1960 + f) - 0.53

  if b < 2 :
    b = b + 0.15 * (2 - b)
  elif b > 20 :
    b = b + 0.22 * (b - 20.1)
  else :
    b = b
  return round(b, 3)

a = hz2bark(65.4)
print(a)
b = hz2bark(600)
print(b)
c = hz2bark(10000)
print(c)