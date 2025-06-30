import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
from collections import defaultdict

class SwarmConsensusModel:
    """
    Cross-inhibition model for robotic swarm consensus using Chemical Reaction Networks (CRN).
    
    Reactions:
    X + Y -> X + U
    X + Y -> U + Y  
    X + U -> X + X
    Y + U -> Y + Y
    X + ZY -> U + ZY
    U + ZY -> Y + ZY
    Y + ZX -> U + ZX
    U + ZX -> X + ZX
    """
    
    def __init__(self, rate_constants=None):
        """Initialize the model with reaction rate constants."""
        if rate_constants is None:
            # Default rate constants for all reactions
            self.rate_constants = {
                'k1': 1.0,  # X + Y -> X + U
                'k2': 1.0,  # X + Y -> U + Y
                'k3': 1.0,  # X + U -> X + X
                'k4': 1.0,  # Y + U -> Y + Y
                'k5': 1.0,  # X + ZY -> U + ZY
                'k6': 1.0,  # U + ZY -> Y + ZY
                'k7': 1.0,  # Y + ZX -> U + ZX
                'k8': 1.0,  # U + ZX -> X + ZX
            }
        else:
            self.rate_constants = rate_constants
    
    def reaction_rates(self, state, t):
        """
        Calculate the reaction rates based on current state.
        state = [X, Y, U, ZX, ZY]
        """
        X, Y, U, ZX, ZY = state
        
        # Ensure non-negative concentrations
        X, Y, U, ZX, ZY = max(0, X), max(0, Y), max(0, U), max(0, ZX), max(0, ZY)
        
        # Reaction rates
        r1 = self.rate_constants['k1'] * X * Y  # X + Y -> X + U
        r2 = self.rate_constants['k2'] * X * Y  # X + Y -> U + Y
        r3 = self.rate_constants['k3'] * X * U  # X + U -> X + X
        r4 = self.rate_constants['k4'] * Y * U  # Y + U -> Y + Y
        r5 = self.rate_constants['k5'] * X * ZY  # X + ZY -> U + ZY
        r6 = self.rate_constants['k6'] * U * ZY  # U + ZY -> Y + ZY
        r7 = self.rate_constants['k7'] * Y * ZX  # Y + ZX -> U + ZX
        r8 = self.rate_constants['k8'] * U * ZX  # U + ZX -> X + ZX
        
        # Differential equations
        dX_dt = -r1 - r2 + r3 - r5 + r8
        dY_dt = -r1 - r2 + r4 - r7 + r6
        dU_dt = r1 + r2 - r3 - r4 + r5 - r6 + r7 - r8
        dZX_dt = 0  # ZX is constant (zealot)
        dZY_dt = 0  # ZY is constant (zealot)
        
        return [dX_dt, dY_dt, dU_dt, dZX_dt, dZY_dt]
    
    def simulate(self, initial_state, time_points):
        """Simulate the system dynamics."""
        solution = odeint(self.reaction_rates, initial_state, time_points)
        return solution
    
    def check_consensus(self, state, m, d):
        """
        Check if consensus is achieved based on the given formula.
        F = (|X + ZX > minm ∧ |X - Y > d|) ∨ |Y + ZY > m ∧ |Y - X > d||)
        """
        X, Y, U, ZX, ZY = state
        
        # Calculate total agents for each decision
        total_X = X + ZX
        total_Y = Y + ZY
        
        # Check consensus conditions
        condition1 = (total_X > m) and (abs(X - Y) > d)
        condition2 = (total_Y > m) and (abs(Y - X) > d)
        
        return condition1 or condition2
    
    def find_consensus_probability(self, N, ZX, ZY, m, d, t, h, num_simulations=1000):
        """
        Find the probability that the system reaches consensus.
        
        Parameters:
        N: Total number of agents
        ZX, ZY: Number of zealots for X and Y
        m: Minimum majority threshold
        d: Minimum difference threshold
        t: Time threshold
        h: Time units to maintain consensus
        num_simulations: Number of Monte Carlo simulations
        """
        consensus_count = 0
        
        for _ in range(num_simulations):
            # Initialize with random distribution of remaining agents
            remaining_agents = N - ZX - ZY
            X_initial = random.randint(0, remaining_agents)
            Y_initial = remaining_agents - X_initial
            U_initial = 0
            
            initial_state = [X_initial, Y_initial, U_initial, ZX, ZY]
            
            # Simulate for time t + h
            time_points = np.linspace(0, t + h, int((t + h) * 10))
            solution = self.simulate(initial_state, time_points)
            
            # Check if consensus is maintained for the last h time units
            consensus_maintained = True
            start_check_idx = int(t * 10)  # Start checking from time t
            
            for i in range(start_check_idx, len(solution)):
                if not self.check_consensus(solution[i], m, d):
                    consensus_maintained = False
                    break
            
            if consensus_maintained:
                consensus_count += 1
        
        return consensus_count / num_simulations

def main():
    """Main function to run the analysis as specified in the objectives."""
    
    # Create the model
    model = SwarmConsensusModel()
    
    print("Robotic Swarm Consensus Analysis")
    print("=" * 50)
    
    # Objective 1: Find probability with specific parameters
    print("\n1. Consensus Probability with Given Parameters:")
    print("   m=50, d=10, t=35, h=40, N=100, ZX=ZY=10")
    print("   Initial: 40 agents type X, 40 agents type Y")
    
    N = 100
    ZX = ZY = 10
    m = 50
    d = 10
    t = 35
    h = 40
    
    prob = model.find_consensus_probability(N, ZX, ZY, m, d, t, h)
    print(f"   Consensus Probability: {prob:.3f}")
    
    # Objective 2: Explore varying ZX and ZY values
    print("\n2. Exploring Different Zealot Distributions:")
    zealot_values = [5, 10, 15, 20, 25]
    zealot_probs = []
    
    for zx in zealot_values:
        for zy in zealot_values:
            if zx + zy < N:  # Ensure valid configuration
                prob = model.find_consensus_probability(N, zx, zy, m, d, t, h, num_simulations=500)
                zealot_probs.append((zx, zy, prob))
                print(f"   ZX={zx}, ZY={zy}: Probability = {prob:.3f}")
    
    # Objective 3: Explore group size effects
    print("\n3. Group Size Effects on Consensus:")
    group_sizes = [50, 100, 150, 200, 250]
    size_probs = []
    
    for n in group_sizes:
        # Keep zealot ratio constant
        zx = zy = int(n * 0.1)  # 10% zealots each
        prob = model.find_consensus_probability(n, zx, zy, int(n*0.5), d, t, h, num_simulations=500)
        size_probs.append((n, prob))
        print(f"   N={n}: Probability = {prob:.3f}")
    
    # Visualization
    create_visualizations(zealot_probs, size_probs, model)

def create_visualizations(zealot_probs, size_probs, model):
    """Create visualizations of the results."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sample trajectory
    initial_state = [40, 40, 0, 10, 10]
    time_points = np.linspace(0, 100, 1000)
    solution = model.simulate(initial_state, time_points)
    
    ax1.plot(time_points, solution[:, 0], label='X agents', linewidth=2)
    ax1.plot(time_points, solution[:, 1], label='Y agents', linewidth=2)
    ax1.plot(time_points, solution[:, 2], label='U agents', linewidth=2)
    ax1.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='ZX zealots')
    ax1.axhline(y=10, color='blue', linestyle='--', alpha=0.7, label='ZY zealots')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Agents')
    ax1.set_title('Sample Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Zealot distribution heatmap
    if zealot_probs:
        zealot_values = sorted(set([zx for zx, zy, prob in zealot_probs]))
        prob_matrix = np.zeros((len(zealot_values), len(zealot_values)))
        
        for zx, zy, prob in zealot_probs:
            i = zealot_values.index(zx)
            j = zealot_values.index(zy)
            prob_matrix[i, j] = prob
        
        im = ax2.imshow(prob_matrix, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(len(zealot_values)))
        ax2.set_yticks(range(len(zealot_values)))
        ax2.set_xticklabels(zealot_values)
        ax2.set_yticklabels(zealot_values)
        ax2.set_xlabel('ZY (Y Zealots)')
        ax2.set_ylabel('ZX (X Zealots)')
        ax2.set_title('Consensus Probability vs Zealot Distribution')
        plt.colorbar(im, ax=ax2)
    
    # 3. Group size effect
    if size_probs:
        sizes, probs = zip(*size_probs)
        ax3.plot(sizes, probs, 'bo-', linewidth=2, markersize=8)
        ax3.set_xlabel('Group Size (N)')
        ax3.set_ylabel('Consensus Probability')
        ax3.set_title('Effect of Group Size on Consensus')
        ax3.grid(True, alpha=0.3)
    
    # 4. Phase space trajectory
    ax4.plot(solution[:, 0], solution[:, 1], linewidth=2)
    ax4.scatter(solution[0, 0], solution[0, 1], color='green', s=100, label='Start', zorder=5)
    ax4.scatter(solution[-1, 0], solution[-1, 1], color='red', s=100, label='End', zorder=5)
    ax4.set_xlabel('X Agents')
    ax4.set_ylabel('Y Agents')
    ax4.set_title('Phase Space Trajectory (X vs Y)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_consensus_formula():
    """Demonstrate the consensus formula with examples."""
    model = SwarmConsensusModel()
    
    print("\nConsensus Formula Demonstration:")
    print("F = (|X + ZX > m ∧ |X - Y > d|) ∨ |Y + ZY > m ∧ |Y - X > d||)")
    print("=" * 60)
    
    test_cases = [
        ([30, 20, 0, 10, 10], 50, 10, "No consensus - neither group dominates"),
        ([45, 15, 0, 10, 10], 50, 10, "X consensus - X+ZX=55>50, |X-Y|=30>10"),
        ([15, 45, 0, 10, 10], 50, 10, "Y consensus - Y+ZY=55>50, |Y-X|=30>10"),
        ([35, 35, 0, 10, 10], 50, 10, "No consensus - difference too small"),
    ]
    
    for state, m, d, description in test_cases:
        consensus = model.check_consensus(state, m, d)
        X, Y, U, ZX, ZY = state
        print(f"State: X={X}, Y={Y}, U={U}, ZX={ZX}, ZY={ZY}")
        print(f"X+ZX={X+ZX}, Y+ZY={Y+ZY}, |X-Y|={abs(X-Y)}")
        print(f"Consensus: {consensus} - {description}")
        print()

if __name__ == "__main__":
    main()
    demonstrate_consensus_formula()
