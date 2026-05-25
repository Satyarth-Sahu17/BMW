# Data Policy and Privacy Guidelines

## Purpose

This document outlines ethical considerations and best practices for handling sensitive data in the BioAcoustic Monitoring of Endangered Wildlife (BMW) project.

## Location Data Privacy

### Risks
- **Poaching**: Precise GPS coordinates of endangered species can be exploited by poachers
- **Human Disturbance**: Public knowledge of animal locations may lead to excessive human activity in sensitive habitats

### Recommendations

1. **Data Masking**
   - Reduce GPS precision (e.g., round to 0.1 degree or grid cells)
   - Add random noise to coordinates for public-facing dashboards
   - Provide precise locations only to authorized researchers and conservation officers

2. **Access Control**
   - Implement role-based access: public, researcher, admin
   - Require authentication for sensitive endpoints
   - Log all access to precise location data

3. **Aggregation**
   - Display region-level statistics instead of point locations
   - Use heat maps with coarse resolution for public visualizations

## Audio Data

### Considerations
- Audio recordings may contain human voices or activities
- Recordings could identify specific locations or individuals

### Guidelines
- Store raw audio securely with restricted access
- Anonymize or remove segments with identifiable human speech before public release
- Document data retention policies and deletion schedules

## Threat Detection

### Alert Distribution
- Gunshot/chainsaw detections should be reported immediately to authorities
- Implement secure, encrypted communication channels for threat alerts
- Avoid public broadcast of real-time threat locations

## Research Use

### Data Sharing
- Provide anonymized datasets for academic research
- Require data use agreements that prohibit misuse
- Include citation requirements and acknowledgments

### Publication
- Follow FAIR data principles (Findable, Accessible, Interoperable, Reusable)
- Redact sensitive information in publications
- Coordinate with conservation organizations before releasing findings

## Compliance

Ensure compliance with:
- Local wildlife protection laws
- Data protection regulations (GDPR, CCPA, etc.)
- Institutional review board (IRB) requirements for research

## Contact

For questions or to report concerns about data handling, contact the project maintainers or your institutional data protection officer.

## Version

Version 1.0 - January 2025
