from typing import Callable

def newton_raphson(
    f: Callable[[float], float], 
    df: Callable[[float], float], 
    x0: float, 
    tolerance: float = 1e-9, 
    max_iterations: int = 100
) -> float:
    """
    Finds the root of a function using the Newton-Raphson method.

    Parameters:
    f (Callable): The function for which to find the root (f(x) = 0).
    df (Callable): The derivative of the function f'(x).
    x0 (float): The initial guess.
    tolerance (float): The acceptable error margin.
    max_iterations (int): The maximum number of iterations.

    Returns:
    float: The approximated root.
    
    Raises:
    ValueError: If the derivative evaluates to zero or max iterations are exceeded.
    """
    x_n = x0
    
    for i in range(max_iterations):
        f_x = f(x_n)
        df_x = df(x_n)
        
        if df_x == 0:
            raise ValueError(f"Derivative is zero at x = {x_n}. The method cannot continue.")
            
        x_next = x_n - (f_x / df_x)
        
        # Check for convergence
        if abs(x_next - x_n) < tolerance:
            return x_next
            
        x_n = x_next
        
    raise ValueError(f"Failed to converge after {max_iterations} iterations.")

# f(T) = 0
def f(T: float) -> float:
    return (T**2.5) / (T + 110.4) - 3767.3488

# f'(T)
def df(T: float) -> float:
    return (T**1.5 * (1.5 * T + 2.5 * 110.4)) / ((T + 110.4)**2)

try:
    root = newton_raphson(f, df, 300)
    print(f"T = {root:.4f}")
except ValueError as e:
    print(f"Error: {e}")