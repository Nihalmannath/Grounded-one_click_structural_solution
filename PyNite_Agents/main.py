from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from vector import retriever


def StructuralPropertiesRetriever(result: str = "structural properties") -> dict:
    """
    This function retrieves structural properties from a vector store.
    It uses the OllamaLLM model to process the retrieved data and format it for structural engineering analysis.
    """
    # Ask user to choose material type
    print("Please choose a material type for structural analysis:")
    print("1. Concrete (C25/30, C30/37, C35/45)")
    print("2. Steel (S235, S275, S355)")
    print("3. Timber")
    print("4. Aluminum")
    
    material_choice = input("Enter your choice (1-4): ").strip()
    
    material_map = {
        "1": "Concrete C30/37",
        "2": "Steel S355", 
        "3": "Timber",
        "4": "Aluminum"
    }
    
    chosen_material = material_map.get(material_choice, "Concrete C30/37")
    print(f"You selected: {chosen_material}")
    
    model = OllamaLLM(model="llama3.2")

    template = """
    You are a structural Engineer doing analysis of systems. 
    You are well aware of the cross sections, materials and their physical properties. 
    You are also aware of the different types of loads and how they affect the structure.

    The user has selected {material_type} as the material for structural analysis.

    Based on this material choice, provide realistic values for these structural properties:

    1. Material Selection and Beam Length:
    material_type = {material_type}
    beam_lengths = [appropriate length in meters]

    2. Material Properties:
    E = [Young's modulus in Pa]
    nu = [Poisson's ratio]
    rho = [density in kg/m³]

    3. Cross Section Properties:
    cross_section_type = [type of cross section appropriate for this material]
    A = [cross-sectional area in m²]
    Iy = [moment of inertia about y-axis in m⁴]
    Iz = [moment of inertia about z-axis in m⁴]
    J = [torsional constant in m⁴]

    Return ONLY the variable assignments in the exact format shown above, with numerical values (no units in the values).
    Each line should be: variable_name = numerical_value
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    llm_result = chain.invoke({"material_type": chosen_material})
    
    # Parse the LLM result and create a dictionary with default values
    properties_dict = {
        "material_type": "concrete",
        "beam_lengths": 5.0,
        "E": 30000000000,
        "nu": 0.2,
        "rho": 2400,
        "cross_section_type": "rectangular",
        "A": 0.01,
        "Iy": 0.0001,
        "Iz": 0.0001,
        "J": 0.0001
    }
    
    # Debug: print the raw LLM result
    print("Raw LLM Result:")
    print(repr(llm_result))
    print("Parsed lines:")
    
    # Try to parse the LLM output for actual values
    if llm_result:
        lines = llm_result.strip().split('\n')
        for i, line in enumerate(lines):
            print(f"Line {i}: {repr(line)}")
            if '=' in line:
                try:
                    key, value = line.split('=', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    # Remove any brackets or quotes from value
                    value = value.strip('[](){}"\' ')
                    
                    print(f"Parsing: key='{key}', value='{value}'")
                    
                    # Map various possible key names to our standard keys
                    key_mapping = {
                        "material_type": "material_type",
                        "beam_lengths": "beam_lengths",
                        "beam_length": "beam_lengths",
                        "e": "E",
                        "young's modulus": "E",
                        "nu": "nu",
                        "poisson's ratio": "nu",
                        "rho": "rho",
                        "density": "rho",
                        "cross_section_type": "cross_section_type",
                        "cross section type": "cross_section_type",
                        "a": "A",
                        "area": "A",
                        "iy": "Iy",
                        "iz": "Iz",
                        "j": "J",
                        "torsional constant": "J"
                    }
                    
                    # Find the matching key
                    mapped_key = key_mapping.get(key)
                    if mapped_key:
                        if mapped_key in ["material_type", "cross_section_type"]:
                            properties_dict[mapped_key] = value.strip('"\'')
                            print(f"Set {mapped_key} = {properties_dict[mapped_key]}")
                        else:
                            # Try to convert to float
                            try:
                                properties_dict[mapped_key] = float(value)
                                print(f"Set {mapped_key} = {properties_dict[mapped_key]}")
                            except ValueError:
                                print(f"Could not convert '{value}' to float for key {mapped_key}")
                    else:
                        print(f"Unknown key: '{key}'")
                        
                except Exception as e:
                    print(f"Error parsing line '{line}': {e}")
                    continue
    
    print("---")
    return properties_dict


# Execute the function
if __name__ == "__main__":
    props_result = StructuralPropertiesRetriever()
    print("Final Dictionary:")
    print(props_result)
    
    # Extract individual variables
    material_type = props_result.get("material_type")
    beam_lengths = props_result.get("beam_lengths")
    E = props_result.get("E")
    nu = props_result.get("nu")
    rho = props_result.get("rho")
    cross_section_type = props_result.get("cross_section_type")
    A = props_result.get("A")
    Iy = props_result.get("Iy")
    Iz = props_result.get("Iz")
    J = props_result.get("J")
    
    print("\nExtracted structural properties:")
    print(f"material_type = {material_type}")
    print(f"beam_lengths = {beam_lengths}")
    print(f"E = {E}")
    print(f"nu = {nu}")
    print(f"rho = {rho}")
    print(f"cross_section_type = {cross_section_type}")
    print(f"A = {A}")
    print(f"Iy = {Iy}")
    print(f"Iz = {Iz}")
    print(f"J = {J}")
