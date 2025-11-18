import React, { useState, useEffect, useMemo } from 'react';
import Graph from 'react-graph-vis';

function GraphViewer({ graphData, analysisMethod }) {
    const [network, setNetwork] = useState(null);
    const [selectedNode, setSelectedNode] = useState(null);
    const [edgeFilter, setEdgeFilter] = useState(50);
    const [minWeight, setMinWeight] = useState(0);
    const [graphStats, setGraphStats] = useState({});
    const [showPhysics, setShowPhysics] = useState(false);

    const filteredGraphData = useMemo(() => {
        if (!graphData || !graphData.nodes || !graphData.edges) return null;

        const sortedEdges = [...graphData.edges].sort((a, b) => 
            Math.abs(b.weight) - Math.abs(a.weight)
        );

        const filtered = sortedEdges
            .filter(edge => Math.abs(edge.weight) >= minWeight)
            .slice(0, edgeFilter);

        const nodeSet = new Set();
        filtered.forEach(edge => {
            nodeSet.add(edge.source);
            nodeSet.add(edge.target);
        });

        const filteredNodes = graphData.nodes.filter(node => nodeSet.has(node));

        return {
            nodes: filteredNodes,
            edges: filtered,
            totalNodes: graphData.nodes.length,
            totalEdges: graphData.edges.length
        };
    }, [graphData, edgeFilter, minWeight]);

    useEffect(() => {
        if (filteredGraphData) {
            const stats = {
                totalNodes: filteredGraphData.totalNodes,
                totalEdges: filteredGraphData.totalEdges,
                displayedNodes: filteredGraphData.nodes.length,
                displayedEdges: filteredGraphData.edges.length,
                avgWeight: filteredGraphData.edges.length > 0 
                    ? filteredGraphData.edges.reduce((sum, edge) => sum + Math.abs(edge.weight), 0) / filteredGraphData.edges.length 
                    : 0
            };
            setGraphStats(stats);
        }
    }, [filteredGraphData]);

    useEffect(() => {
        if (network && !showPhysics) {
            network.setOptions({ physics: false });
        }
    }, [network, showPhysics]);

    if (!filteredGraphData) {
        return (
            <div className="no-data">
                <div className="no-data-icon">📊</div>
                <h3>No Analysis Data</h3>
                <p>Upload a CSV file and run analysis to see the causal graph</p>
            </div>
        );
    }

    const { nodes, edges } = filteredGraphData;

    const transformedNodes = nodes.map(node => ({
        id: node,
        label: node,
        title: node
    }));

    const transformedLinks = edges.map((link, idx) => {
        const weightAbs = Math.abs(link.weight);
    
        return {
            id: `edge-${idx}`,
            from: link.source,
            to: link.target,
            label: weightAbs.toFixed(2),
    
            // Proper vis-network syntax
            width: Math.min(1 + weightAbs * 2, 8),
    
            font: {
                size: 18,
                color: "white",
                strokeWidth: 4,
                strokeColor: "#000000",
                face: "arial",
            },
    
            title: `Weight: ${link.weight.toFixed(3)}
    P-value: ${link.p_value?.toExponential(2) || "N/A"}
    Lag: ${link.lag || "N/A"}`
        };
    });
    
    

    const graph = {
        nodes: transformedNodes,
        edges: transformedLinks
    };

    const options = {
        layout: {
          hierarchical: false,
          improvedLayout: true
        },
      
        edges: {
          color: {
            color: "#7FB3FF",
            highlight: "#A4C8FF",
            hover: "#A4C8FF"
          },
          arrows: {
            to: {
              enabled: true,
              scaleFactor: 1.4
            }
          },
          width: 2,
          smooth: {
            enabled: true,
            type: "continuous"
          },
          font: {
            size: 18,
            color: "#FFFFFF",
            strokeWidth: 4,
            strokeColor: "#000000",
            align: "horizontal",
            face: "arial"
          }
        },
      
        nodes: {
          shape: "dot",
          size: 35,
          borderWidth: 3,
          color: {
            background: "#6FCF6F",
            border: "#2E7D32",
            highlight: {
              background: "#8BFF8B",
              border: "#3CAA3C"
            },
            hover: {
              background: "#8BFF8B",
              border: "#3CAA3C"
            }
          },
          font: {
            size: 20,
            color: "#FFFFFF",
            face: "arial"
          }
        },
      
        physics: {
          enabled: showPhysics,
          solver: "forceAtlas2Based",
          forceAtlas2Based: {
            gravitationalConstant: -40,
            centralGravity: 0.01,
            springConstant: 0.12,
            springLength: 140,
            avoidOverlap: 0.7
          }
        },
      
        interaction: {
          dragNodes: true,
          dragView: true,
          zoomView: true,
          hover: true,
          tooltipDelay: 100
        }
      };
          

    const events = {
        stabilized: () => {
            if (network && showPhysics) {
                network.setOptions({ physics: false });
                setShowPhysics(false);
            }
        },
        click: (params) => {
            if (params.nodes.length > 0) {
                setSelectedNode(params.nodes[0]);
            } else {
                setSelectedNode(null);
            }
        }
    };

    return (
        <div className="graph-viewer">
            <div className="graph-header">
                <div className="graph-title">
                    <h3>{analysisMethod} Analysis</h3>
                    <div className="graph-controls">
                        <button 
                            className="control-btn"
                            onClick={() => network?.fit()}
                            title="Fit to view"
                        >
                            Fit View
                        </button>
                        <button 
                            className="control-btn"
                            onClick={() => {
                                setShowPhysics(true);
                                network?.stabilize();
                            }}
                            title="Rearrange"
                        >
                            Rearrange
                        </button>
                    </div>
                </div>
            </div>

            <div className="graph-stats">
                <div className="stat">
                    <span className="stat-label">Total Nodes:</span>
                    <span className="stat-value">{graphStats.totalNodes}</span>
                </div>
                <div className="stat">
                    <span className="stat-label">Total Edges:</span>
                    <span className="stat-value">{graphStats.totalEdges}</span>
                </div>
                <div className="stat">
                    <span className="stat-label">Displayed Nodes:</span>
                    <span className="stat-value highlight">{graphStats.displayedNodes}</span>
                </div>
                <div className="stat">
                    <span className="stat-label">Displayed Edges:</span>
                    <span className="stat-value highlight">{graphStats.displayedEdges}</span>
                </div>
                <div className="stat">
                    <span className="stat-label">Avg Weight:</span>
                    <span className="stat-value">{graphStats.avgWeight?.toFixed(2) || 'N/A'}</span>
                </div>
            </div>

            <div className="filter-controls">
                <div className="filter-group">
                    <label>Show Top N Edges: {edgeFilter}</label>
                    <input
                        type="range"
                        min="10"
                        max={Math.min(graphStats.totalEdges || 500, 500)}
                        value={edgeFilter}
                        onChange={(e) => setEdgeFilter(Number(e.target.value))}
                    />
                </div>
                <div className="filter-group">
                    <label>Min Weight: {minWeight.toFixed(2)}</label>
                    <input
                        type="range"
                        min="0"
                        max="5"
                        step="0.1"
                        value={minWeight}
                        onChange={(e) => setMinWeight(Number(e.target.value))}
                    />
                </div>
            </div>

            {graphStats.totalEdges > edgeFilter && (
                <div className="warning-banner">
                    ⚠️ Large dataset detected. Showing top {edgeFilter} of {graphStats.totalEdges} edges. 
                    Adjust filters above to see different edges.
                </div>
            )}

            {selectedNode && (
                <div className="node-info">
                    <strong>Selected Node:</strong> {selectedNode}
                </div>
            )}
            
            <div className="graph-container">
                <Graph
                    key={`${edgeFilter}-${minWeight}`}
                    graph={graph}
                    options={options}
                    events={events}
                    getNetwork={setNetwork}
                    style={{ height: "600px" }}
                />
            </div>
        </div>
    );
}

export default GraphViewer;
