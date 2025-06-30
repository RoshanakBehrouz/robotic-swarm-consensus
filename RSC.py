import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class CrossInhibitionModel:
    def __init__(self, N=100, ZX=10, ZY=10, X0=40, Y0=40, qX=1.0, qY=1.0):
        """
        Initialize the cross-inhibition model.
        
        Parameters:
        - N: Total number of agents
        - ZX: Number of X zealots
        - ZY: Number of Y zealots
        - X0: Initial number of X agents
        - Y0: Initial number of Y agents
        - qX: Rate parameter for X reactions
        - qY: Rate parameter for Y reactions
        """
        # Validate initial conditions
        assert X0 + Y0 + ZX + ZY <= N, "Initial conditions exceed total agents"
        
        self.N = N
        self.U0 = N - (X0 + Y0 + ZX + ZY)  # Calculate initial U agents
        
        # Current state
        self.state = {
            'X': X0,
            'Y': Y0,
            'U': self.U0,
            'ZX': ZX,
            'ZY': ZY
        }
        
        # Rate parameters
        self.qX = qX
        self.qY = qY
        
        # Time tracking
        self.time = 0.0
        self.history = []
        self.record_state()
        
    def record_state(self):
        """Record the current state and time."""
        self.history.append({
            'time': self.time,
            'X': self.state['X'],
            'Y': self.state['Y'],
            'U': self.state['U'],
            'ZX': self.state['ZX'],
            'ZY': self.state['ZY']
        })
    
    def calculate_propensities(self):
        """Calculate propensities for all reactions."""
        X, Y, U, ZX, ZY = self.state['X'], self.state['Y'], self.state['U'], self.state['ZX'], self.state['ZY']
        
        propensities = {
            'r1': self.qX * X * Y,      # X + Y → X + U
            'r2': self.qY * X * Y,      # X + Y → U + Y
            'r3': self.qX * X * U,      # X + U → X + X
            'r4': self.qY * Y * U,      # Y + U → Y + Y
            'r5': self.qY * X * ZY,     # X + ZY → U + ZY
            'r6': self.qY * U * ZY,     # U + ZY → Y + ZY
            'r7': self.qX * Y * ZX,     # Y + ZX → U + ZX
            'r8': self.qX * U * ZX      # U + ZX → X + ZX
        }
        
        return propensities
    
    def execute_reaction(self, reaction):
        """Update state based on the selected reaction."""
        if reaction == 'r1':   # X + Y → X + U
            self.state['Y'] -= 1
            self.state['U'] += 1
        elif reaction == 'r2': # X + Y → U + Y
            self.state['X'] -= 1
            self.state['U'] += 1
        elif reaction == 'r3': # X + U → X + X
            self.state['U'] -= 1
            self.state['X'] += 1
        elif reaction == 'r4': # Y + U → Y + Y
            self.state['U'] -= 1
            self.state['Y'] += 1
        elif reaction == 'r5': # X + ZY → U + ZY
            self.state['X'] -= 1
            self.state['U'] += 1
        elif reaction == 'r6': # U + ZY → Y + ZY
            self.state['U'] -= 1
            self.state['Y'] += 1
        elif reaction == 'r7': # Y + ZX → U + ZX
            self.state['Y'] -= 1
            self.state['U'] += 1
        elif reaction == 'r8': # U + ZX → X + ZX
            self.state['U'] -= 1
            self.state['X'] += 1
    
    def step(self):
        """Execute one step of the Gillespie algorithm."""
        # Calculate all reaction propensities
        propensities = self.calculate_propensities()
        total_propensity = sum(propensities.values())
        
        if total_propensity <= 0:
            return False  # No more reactions possible
        
        # Determine time until next reaction
        tau = np.random.exponential(1.0 / total_propensity)
        self.time += tau
        
        # Select which reaction occurs
        r = random.random() * total_propensity
        cumulative = 0.0
        for reaction, propensity in propensities.items():
            cumulative += propensity
            if r <= cumulative:
                self.execute_reaction(reaction)
                break
        
        # Record the new state
        self.record_state()
        return True
    
    def simulate(self, max_time):
        """Run simulation until max_time is reached."""
        while self.time < max_time and self.step():
            pass
    
    def check_consensus(self, t, h, m, d):
        """
        Check if consensus was reached according to the specified property.
        
        Parameters:
        - t: Time before which consensus must be reached
        - h: Duration consensus must be maintained
        - m: Majority parameter (percentage)
        - d: Minimum difference between groups
        
        Returns:
        - 1 if X consensus reached
        - -1 if Y consensus reached
        - 0 if no consensus reached
        """
        min_m = (m / 100) * self.N
        consensus_start = None
        current_consensus = 0
        
        # Find all time points before t
        time_points = [entry for entry in self.history if entry['time'] <= t]
        
        for entry in time_points:
            X_total = entry['X'] + entry['ZX']
            Y_total = entry['Y'] + entry['ZY']
            
            # Check for X consensus
            if X_total > min_m and (X_total - Y_total) > d:
                if current_consensus != 1:
                    consensus_start = entry['time']
                    current_consensus = 1
            # Check for Y consensus
            elif Y_total > min_m and (Y_total - X_total) > d:
                if current_consensus != -1:
                    consensus_start = entry['time']
                    current_consensus = -1
            else:
                # Consensus lost
                consensus_start = None
                current_consensus = 0
            
            # Check if consensus has been maintained for h time units
            if consensus_start is not None and (entry['time'] - consensus_start) >= h:
                return current_consensus
        
        return 0

def run_experiments(num_simulations=100, N=100, ZX=10, ZY=10, X0=40, Y0=40, 
                    t=35, h=40, m=50, d=10):
    """Run multiple simulations to estimate consensus probability."""
    results = {'X': 0, 'Y': 0, 'none': 0}
    
    for _ in range(num_simulations):
        model = CrossInhibitionModel(N=N, ZX=ZX, ZY=ZY, X0=X0, Y0=Y0)
        model.simulate(max_time=t + h)  # Simulate slightly longer than needed
        
        consensus = model.check_consensus(t, h, m, d)
        
        if consensus == 1:
            results['X'] += 1
        elif consensus == -1:
            results['Y'] += 1
        else:
            results['none'] += 1
    
    return {k: v / num_simulations for k, v in results.items()}

def vary_zealots(max_zealots=40, num_simulations=100, N=100, X0=40, Y0=40, 
                 t=35, h=40, m=50, d=10):
    """Explore how zealot count affects consensus probability."""
    zealot_counts = range(0, max_zealots + 1, 2)
    results = []
    
    for z in zealot_counts:
        ZX = ZY = z
        # Adjust initial X and Y counts to maintain total agents
        adjusted_X0 = X0 - (z - 10) if z > 10 else X0
        adjusted_Y0 = Y0 - (z - 10) if z > 10 else Y0
        
        # Ensure we don't have negative counts
        adjusted_X0 = max(0, adjusted_X0)
        adjusted_Y0 = max(0, adjusted_Y0)
        
        probs = run_experiments(num_simulations=num_simulations, N=N, 
                               ZX=ZX, ZY=ZY, X0=adjusted_X0, Y0=adjusted_Y0,
                               t=t, h=h, m=m, d=d)
        results.append((z, probs))
    
    return results

def vary_group_size(group_sizes, num_simulations=100, zealot_ratio=0.1, X_ratio=0.4, 
                    Y_ratio=0.4, t=35, h=40, m=50, d=10):
    """Explore how group size affects consensus probability."""
    results = []
    
    for N in group_sizes:
        ZX = ZY = int(N * zealot_ratio / 2)
        X0 = int(N * X_ratio)
        Y0 = int(N * Y_ratio)
        
        probs = run_experiments(num_simulations=num_simulations, N=N, 
                              ZX=ZX, ZY=ZY, X0=X0, Y0=Y0,
                              t=t, h=h, m=m, d=d)
        results.append((N, probs))
    
    return results

def plot_zealot_results(results):
    """Plot results from zealot variation experiment."""
    z_counts = [r[0] for r in results]
    x_probs = [r[1]['X'] for r in results]
    y_probs = [r[1]['Y'] for r in results]
    none_probs = [r[1]['none'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(z_counts, x_probs, 'b-', label='X consensus')
    plt.plot(z_counts, y_probs, 'r-', label='Y consensus')
    plt.plot(z_counts, none_probs, 'g-', label='No consensus')
    
    plt.xlabel('Number of Zealots (ZX = ZY)')
    plt.ylabel('Probability')
    plt.title('Effect of Zealot Count on Consensus Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_group_size_results(results):
    """Plot results from group size variation experiment."""
    sizes = [r[0] for r in results]
    x_probs = [r[1]['X'] for r in results]
    y_probs = [r[1]['Y'] for r in results]
    none_probs = [r[1]['none'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, x_probs, 'b-', label='X consensus')
    plt.plot(sizes, y_probs, 'r-', label='Y consensus')
    plt.plot(sizes, none_probs, 'g-', label='No consensus')
    
    plt.xlabel('Group Size (N)')
    plt.ylabel('Probability')
    plt.title('Effect of Group Size on Consensus Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main analysis
if __name__ == "__main__":
    # Base case from the problem statement
    print("Running base case simulations...")
    base_results = run_experiments(num_simulations=500)
    print("\nBase case results (m=50, d=10, t=35, h=40, N=100, ZX=ZY=10):")
    print(f"X consensus probability: {base_results['X']:.3f}")
    print(f"Y consensus probability: {base_results['Y']:.3f}")
    print(f"No consensus probability: {base_results['none']:.3f}")
    
    # Vary zealot counts
    print("\nRunning zealot variation experiments...")
    zealot_results = vary_zealots(max_zealots=40, num_simulations=200)
    plot_zealot_results(zealot_results)
    
    # Vary group size
    print("\nRunning group size variation experiments...")
    group_sizes = range(20, 501, 20)
    group_results = vary_group_size(group_sizes, num_simulations=200)
    plot_group_size_results(group_results)
