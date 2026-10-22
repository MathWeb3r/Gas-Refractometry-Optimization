import json
import warnings
from pathlib import Path
from typing import List


# Relative path to database.json
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / 'data' / 'gas_database.json'


class GasDatabase:
    """
    Manager and query interface for the gas refractivity database (gas_database.json).

    This class centralizes access to Sellmeier dispersion coefficients, spectral
    validity ranges, and metrological uncertainties. It provides caching for Gas
    instances, syntax validation methods (static), and dynamic equation selection
    based on validity range and measurement uncertainty.
    """

    def __init__(self, path=DEFAULT_DB_PATH):
        self.path = Path(path)

        # Load database.json file
        with open(self.path, 'r', encoding='utf-8') as f:
            self._raw_data = json.load(f)

        # Instance cache to avoid redundant Gas object creation
        self._gas_cache = {}

    #
    # Static methods (Syntax & Format Validations)
    #

    @staticmethod
    def is_generic(gasid: str) -> bool:
        """
        Check whether an ID is generic or specific (indexed).

        E.g., 'co2' is generic, while 'co2_1' is specific (non-generic).

        Parameters
        ----------
        gasid : str
            Gas identifier to evaluate.

        Returns
        -------
        bool
            True if the ID is generic (does not contain '_'), False otherwise.
        """
        return isinstance(gasid, str) and ('_' not in gasid.strip())

    @staticmethod
    def all_generic(idlist: List[str]) -> bool:
        """
        Check if all identifiers in a list are generic.

        Parameters
        ----------
        idlist : List[str]
            List of gas identifiers.

        Returns
        -------
        bool
            True if all elements are generic, False otherwise.
        """
        return all(GasDatabase.is_generic(gid) for gid in idlist)

    @staticmethod
    def all_nongeneric(idlist: List[str]) -> bool:
        """
        Check if all identifiers in a list are specific (non-generic).

        Parameters
        ----------
        idlist : List[str]
            List of gas identifiers.

        Returns
        -------
        bool
            True if all elements contain an index delimiter ('_'), False otherwise.
        """
        return all(not GasDatabase.is_generic(gid) for gid in idlist)

    @staticmethod
    def validate_wavelengths(wvlens: list) -> bool:
        """
        Validate that wavelengths are non-empty and strictly positive.

        Parameters
        ----------
        wvlens : list
            Sequence of wavelengths in micrometers.

        Returns
        -------
        bool
            True if valid.

        Raises
        -------
        ValueError
            If the wavelength list is empty or contains non-positive values.
        """
        if wvlens is None or len(wvlens) == 0:
            raise ValueError("Wavelength list must not be empty.")

        if any(w <= 0 for w in wvlens):
            raise ValueError("All wavelengths must be strictly positive (> 0).")

        return True

    #
    # Non-static methods (Database Queries & Semantics)
    #

    def in_database(self, gasid: str) -> bool:
        """
        Verify if a specific gas ID exists in the database.

        Parameters
        ----------
        gasid : str
            Specific gas identifier (e.g., 'co2_1').

        Returns
        -------
        bool
            True if the ID exists in the database.

        Raises
        ------
        KeyError
            If the gas ID is not found.
        """
        gasid_clean = gasid.lower().strip()
        if gasid_clean not in self._raw_data:
            available = list(self._raw_data.keys())
            raise KeyError(f"Gas ID '{gasid}' not found in database. Available IDs: {available}")
        return True

    def get_gas(self, gas_id: str):
        """
        Retrieve a cached Gas instance for a given specific ID.

        Parameters
        ----------
        gas_id : str
            Specific gas ID (e.g., 'co2_1').

        Returns
        -------
        Gas
            Instantiated Gas object.
        """
        gas_id_clean = gas_id.lower().strip()
        self.in_database(gas_id_clean)

        if gas_id_clean not in self._gas_cache:
            from gasopt.dispersion import Gas
            self._gas_cache[gas_id_clean] = Gas(id=gas_id_clean)

        return self._gas_cache[gas_id_clean]

    def is_valid_ids_list(self, idlist: List[str]) -> bool:
        """
        Validate that the provided identifiers are either all generic or all specific,
        and verify that they exist in the database.

        Parameters
        ----------
        idlist : List[str]
            List of gas identifiers.

        Returns
        -------
        bool
            True if the list is valid.

        Raises
        ------
        ValueError
            If the list is empty or mixes generic and specific identifiers.
        KeyError
            If any specific identifier is missing from the database.
        """
        if not idlist:
            raise ValueError("The gas ID list must not be empty.")

        # Sanitize list (lowercase and strip whitespace)
        sanitized = [str(gid).lower().strip() for gid in idlist]

        is_all_gen = self.all_generic(sanitized)
        is_all_nongen = self.all_nongeneric(sanitized)

        if not (is_all_gen or is_all_nongen):
            raise ValueError(
                "Gas identifiers must be either all generic or all specific (non-generic). "
                f"Received mixed input: {idlist}. Expected format: ['n2', 'co2'] or ['n2_1', 'co2_2']."
            )

        # Check existence in database
        if is_all_nongen:
            for gid in sanitized:
                self.in_database(gid)
        else:
            for gname in sanitized:
                # Raises ValueError if no candidate equations exist for the generic name
                self._get_candidates(gname)

        return True

    def _get_candidates(self, gas_name: str) -> list:
        """
        Retrieve all available equations for a generic gas name.

        E.g., 'co2' -> [Gas('co2_1'), Gas('co2_2'), Gas('co2_3')].

        Parameters
        ----------
        gas_name : str
            Generic gas name (e.g., 'co2', 'n2').

        Returns
        -------
        list
            List of Gas instances registered under this base name.

        Raises
        ------
        ValueError
            If no equations are registered for this gas name.
        """
        gas_name_clean = gas_name.lower().strip()
        candidates = []
        i = 1
        while True:
            curr_id = f"{gas_name_clean}_{i}"
            if curr_id in self._raw_data:
                candidates.append(self.get_gas(curr_id))
                i += 1
            else:
                break

        if not candidates:
            raise ValueError(f"No dispersion equations registered for generic gas '{gas_name}'.")

        return candidates

    def get_best_equation(self, gas_name: str, wvlen: float, verbose: bool = False):
        """
        Select the best dispersion equation for a given gas at a specific wavelength.

        The selection filters candidate equations by their validity range [min, max]
        in micrometers. In case of multiple valid candidates, it selects the equation
        with the smallest normalized uncertainty (sigma * multiplier). If no equation
        covers the wavelength, a warning is optionally issued and the first candidate is used.

        Parameters
        ----------
        gas_name : str
            Generic gas name (e.g., 'co2').
        wvlen : float
            Wavelength in micrometers (μm).
        verbose : bool, default=False
            Whether to issue a warning if the wavelength falls outside all validity ranges.

        Returns
        -------
        Gas
            The chosen Gas instance.
        """
        candidates = self._get_candidates(gas_name)

        # Filter equations that cover the requested wavelength
        valid = [g for g in candidates if g.range[0] <= wvlen <= g.range[1]]

        if valid:
            # Tie-break by minimum metrological uncertainty (sigma * multiplier)
            return min(valid, key=lambda g: getattr(g, 'sigma', 0.0) * getattr(g, 'multiplier', 1.0))

        # If none cover the wavelength, warn and extrapolate using the first candidate
        if verbose:
            warnings.warn(
                f"Wavelength {wvlen:.4f} μm is outside the validity range for '{gas_name}'. "
                f"Extrapolating using {candidates[0]}."
            )

        return candidates[0]