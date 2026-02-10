import requests
import time

class FoldseekClient:
    BASE_URL = "https://search.foldseek.com/api"

    def __init__(self):
        self.session = requests.Session()

    def _extract_atom_records(self, pdb_content):
        lines = pdb_content.splitlines()
        first_chain = None
        atoms = []

        for line in lines:
            if line.startswith("ATOM"):
                chain = line[21:22].strip()
                if first_chain is None:
                    first_chain = chain
                if chain != first_chain:
                    break
                atoms.append(line)

        return "\n".join(atoms)

    def submit_search(self, pdb_file_path, databases, mode="3diaa"):
        with open(pdb_file_path) as f:
            pdb_content = f.read()

        filtered_pdb = self._extract_atom_records(pdb_content)

        data = f"q={requests.utils.quote(filtered_pdb)}&mode={mode}"
        for db in databases:
            data += f"&database[]={db}"

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        r = self.session.post(f"{self.BASE_URL}/ticket", data=data, headers=headers)
        r.raise_for_status()
        return r.json()

    def wait_for_job(self, ticket_id, interval=1):
        while True:
            r = self.session.get(f"{self.BASE_URL}/ticket/{ticket_id}")
            r.raise_for_status()
            job = r.json()
            if job["status"] == "COMPLETE":
                return job
            time.sleep(interval)

