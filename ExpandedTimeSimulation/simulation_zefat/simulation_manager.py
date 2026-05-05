
from ExpandedTimeSimulation.simulation_zefat.strategies import DynamicPricingStrategy, ZeroPricingStrategy,AlternativePricingStrategy,MedianPricingStrategy
class SimulationManager:
    def __init__(self):
        self.strategies = {
            "Transport-Adapted": DynamicPricingStrategy(),
            "Online": AlternativePricingStrategy(),
            "Zero Pricing": ZeroPricingStrategy(),
            "Static Median": MedianPricingStrategy()
        }
       
