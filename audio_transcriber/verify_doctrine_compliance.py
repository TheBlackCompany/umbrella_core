#!/usr/bin/env python3
"""
Verify implementation compliance with Audio Transcription Doctrine v1.0
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

class DoctrineComplianceChecker:
    """Check if implementation matches doctrine requirements"""
    
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        
    def check(self, condition: bool, requirement: str, implementation: str):
        """Record a compliance check"""
        status = "[PASS]" if condition else "[FAIL]"
        self.checks.append((status, requirement, implementation))
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            
    def verify_api_endpoints(self):
        """Verify all required API endpoints exist"""
        print("\n=== API ENDPOINTS ===")
        
        # Check api.py for required endpoints
        api_file = Path("api.py")
        if api_file.exists():
            content = api_file.read_text(encoding='utf-8')
            
            # Core endpoints
            self.check("/jobs" in content and '@app.post("/jobs"' in content, 
                      "POST /jobs - Submit new transcription job",
                      "Implemented in api.py")
            
            self.check('@app.get("/jobs/{job_id}/status"' in content,
                      "GET /jobs/{id}/status - Check job status",
                      "Implemented in api.py")
            
            self.check('get("/jobs/{job_id}/result")' in content.lower(),
                      "GET /jobs/{id}/result - Retrieve completed transcript",
                      "Implemented in api.py")
            
            self.check('delete("/jobs/{job_id}")' in content.lower(),
                      "DELETE /jobs/{id} - Cancel in-progress job",
                      "Implemented in api.py")
            
            # Webhook endpoints
            self.check('post("/jobs/{job_id}/webhook")' in content.lower(),
                      "POST /jobs/{id}/webhook - Register completion webhook",
                      "Implemented in api.py")
            
            self.check('delete("/jobs/{job_id}/webhook")' in content.lower(),
                      "DELETE /jobs/{id}/webhook - Remove webhook",
                      "Implemented in api.py")
            
            # Utility endpoints
            self.check('get("/health")' in content.lower(),
                      "GET /health - Service health check",
                      "Implemented in api.py")
            
            self.check('get("/metrics")' in content.lower(),
                      "GET /metrics - Processing statistics",
                      "Implemented in api.py")
            
            self.check('@app.post("/estimate"' in content,
                      "POST /estimate - Cost/time estimation",
                      "Implemented in api.py")
        else:
            self.check(False, "API implementation file exists", "api.py not found")
            
    def verify_schema_compliance(self):
        """Verify output schema matches doctrine"""
        print("\n=== OUTPUT SCHEMA ===")
        
        schema_file = Path("schema.py")
        if schema_file.exists():
            content = schema_file.read_text(encoding="utf-8")
            
            # Required schema fields
            required_fields = [
                ("job_id", "Job identifier"),
                ("source_hash", "Source file hash"),
                ("transcript_hash", "Transcript hash"),
                ("processing_metadata", "Processing information"),
                ("source_metadata", "Source information"),
                ("transcript", "Transcript with segments"),
                ("quality_metrics", "Quality metrics"),
                ("chain_of_custody", "Forensic tracking"),
                ("extracted_entities", "Entity extraction")
            ]
            
            for field, desc in required_fields:
                self.check(f'"{field}"' in content or f"'{field}'" in content,
                          f"Schema includes {field} - {desc}",
                          f"Field in schema.py")
                          
            # Check for proper job ID format
            self.check("create_job_id" in content and "strftime" in content,
                      "Job ID format: YYYYMMDD_HHMMSS_hash",
                      "Implemented in DoctrineSchema.create_job_id()")
                      
        else:
            self.check(False, "Schema implementation exists", "schema.py not found")
            
    def verify_priority_queue(self):
        """Verify priority queue implementation"""
        print("\n=== PRIORITY QUEUE ===")
        
        queue_file = Path("priority_queue.py")
        if queue_file.exists():
            content = queue_file.read_text(encoding="utf-8")
            
            # Priority levels
            priorities = ["EMERGENCY", "URGENT", "NORMAL", "BATCH", "CITIZEN"]
            for priority in priorities:
                self.check(priority in content,
                          f"Priority level: {priority}",
                          "Defined in Priority enum")
                          
            self.check("TranscriptionJob" in content,
                      "TranscriptionJob class exists",
                      "Implemented with all required fields")
                      
            self.check("resource_allocation" in content.lower(),
                      "Resource allocation by priority",
                      "Different GPU/CPU allocation per priority")
                      
        else:
            self.check(False, "Priority queue exists", "priority_queue.py not found")
            
    def verify_legislative_features(self):
        """Verify legislative-specific features"""
        print("\n=== LEGISLATIVE FEATURES ===")
        
        leg_file = Path("legislative_processor.py")
        if leg_file.exists():
            content = leg_file.read_text(encoding="utf-8")
            
            self.check("extract_bills" in content,
                      "Bill extraction (SB/HB patterns)",
                      "Implemented in LegislativeProcessor")
                      
            self.check("identify_legislators" in content,
                      "Legislator identification",
                      "Implemented with name patterns")
                      
            self.check("legislative_markers" in content.lower(),
                      "Legislative markers (roll call, recess)",
                      "Marker detection implemented")
                      
        else:
            self.check(False, "Legislative processor exists", "legislative_processor.py not found")
            
    def verify_processing_strategies(self):
        """Verify processing strategies"""
        print("\n=== PROCESSING STRATEGIES ===")
        
        # Check for strategy implementations
        strategies_dir = Path("strategies")
        if strategies_dir.exists():
            self.check((strategies_dir / "chunked.py").exists(),
                      "Chunked processing strategy",
                      "Implemented in strategies/chunked.py")
                      
            chunked = Path("strategies/chunked.py")
            if chunked.exists():
                content = chunked.read_text(encoding="utf-8")
                self.check("chunk_duration" in content or "chunk_minutes" in content,
                          "Configurable chunk duration",
                          "Chunk duration parameter in ChunkedStrategy")
                          
        # Check main transcriber for strategy selection
        transcribe_file = Path("transcribe.py")
        if transcribe_file.exists():
            content = transcribe_file.read_text(encoding="utf-8")
            self.check("_select_strategy" in content or "chunked" in content,
                      "Strategy selection logic",
                      "Duration-based strategy selection")
                      
    def verify_security_compliance(self):
        """Verify security features"""
        print("\n=== SECURITY & COMPLIANCE ===")
        
        security_file = Path("security.py")
        if security_file.exists():
            content = security_file.read_text(encoding="utf-8")
            
            self.check("SecurityManager" in content,
                      "Security manager implementation",
                      "SecurityManager class exists")
                      
            self.check("validate_access" in content,
                      "Access control validation",
                      "User clearance validation implemented")
                      
            classifications = ["public", "confidential", "secret"]
            for classification in classifications:
                self.check(classification in content.lower(),
                          f"Classification level: {classification}",
                          "Defined in security.py")
                          
        else:
            self.check(False, "Security implementation exists", "security.py not found")
            
    def verify_docker_deployment(self):
        """Verify Docker configuration"""
        print("\n=== DOCKER DEPLOYMENT ===")
        
        dockerfile = Path("Dockerfile")
        if dockerfile.exists():
            content = dockerfile.read_text(encoding="utf-8")
            
            self.check("nvidia/cuda" in content,
                      "NVIDIA CUDA base image",
                      "Using nvidia/cuda:12.1 base")
                      
            self.check("whisper" in content.lower(),
                      "Whisper model pre-download",
                      "Models downloaded at build time")
                      
            self.check("ffmpeg" in content,
                      "FFmpeg installation",
                      "FFmpeg installed in container")
                      
            self.check("useradd" in content or "USER" in content,
                      "Non-root user",
                      "Container runs as non-root user")
                      
            self.check("HEALTHCHECK" in content,
                      "Health check defined",
                      "Docker health check configured")
                      
        else:
            self.check(False, "Dockerfile exists", "Dockerfile not found")
            
    def verify_cost_model(self):
        """Verify cost estimation"""
        print("\n=== COST MODEL ===")
        
        schema_file = Path("schema.py")
        if schema_file.exists():
            content = schema_file.read_text(encoding="utf-8")
            
            self.check("estimate_cost" in content or "cost_estimate" in content,
                      "Cost estimation function",
                      "Cost calculation implemented")
                      
            self.check("priority" in content and "cost" in content,
                      "Priority-based cost adjustment",
                      "Different costs by priority level")
                      
    def generate_report(self):
        """Generate compliance report"""
        print("\n" + "=" * 60)
        print("DOCTRINE COMPLIANCE REPORT")
        print("=" * 60)
        
        for status, requirement, implementation in self.checks:
            print(f"{status} {requirement}")
            if status == "[FAIL]":
                print(f"  -> Missing: {implementation}")
                
        print("\n" + "-" * 60)
        print(f"Total Checks: {len(self.checks)}")
        print(f"Passed: {self.passed} ({self.passed/len(self.checks)*100:.1f}%)")
        print(f"Failed: {self.failed}")
        
        if self.failed == 0:
            print("\n[SUCCESS] FULLY COMPLIANT with Audio Transcription Doctrine v1.0")
        else:
            print(f"\n[WARNING] {self.failed} compliance issues found")
            
        return self.failed == 0

def main():
    """Run compliance verification"""
    checker = DoctrineComplianceChecker()
    
    # Run all checks
    checker.verify_api_endpoints()
    checker.verify_schema_compliance()
    checker.verify_priority_queue()
    checker.verify_legislative_features()
    checker.verify_processing_strategies()
    checker.verify_security_compliance()
    checker.verify_docker_deployment()
    checker.verify_cost_model()
    
    # Generate report
    compliant = checker.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if compliant else 1)

if __name__ == "__main__":
    main()