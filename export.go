package hdbscan

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
)

// ExportFormat identifies a streaming tree encoding.
type ExportFormat string

const (
	ExportCSV  ExportFormat = "csv"
	ExportJSON ExportFormat = "json"
)

// WriteMST streams the minimum spanning tree to writer.
func (result Result) WriteMST(writer io.Writer, format ExportFormat) error {
	return writeRecords(writer, format, []string{"from", "to", "distance"}, len(result.MinimumSpanningTree), func(i int) []string {
		edge := result.MinimumSpanningTree[i]
		return []string{strconv.Itoa(edge.From), strconv.Itoa(edge.To), strconv.FormatFloat(edge.Distance, 'g', -1, 64)}
	}, func(encoder *json.Encoder, i int) error { return encoder.Encode(result.MinimumSpanningTree[i]) })
}

// WriteSingleLinkageTree streams the single-linkage hierarchy to writer.
func (result Result) WriteSingleLinkageTree(writer io.Writer, format ExportFormat) error {
	return writeRecords(writer, format, []string{"left", "right", "distance", "size"}, len(result.SingleLinkageTree), func(i int) []string {
		edge := result.SingleLinkageTree[i]
		return []string{strconv.Itoa(edge.Left), strconv.Itoa(edge.Right), strconv.FormatFloat(edge.Distance, 'g', -1, 64), strconv.Itoa(edge.Size)}
	}, func(encoder *json.Encoder, i int) error { return encoder.Encode(result.SingleLinkageTree[i]) })
}

// WriteCondensedTree streams the condensed hierarchy to writer.
func (result Result) WriteCondensedTree(writer io.Writer, format ExportFormat) error {
	return writeRecords(writer, format, []string{"parent", "child", "lambda", "child_size"}, len(result.CondensedTree), func(i int) []string {
		edge := result.CondensedTree[i]
		return []string{strconv.Itoa(edge.Parent), strconv.Itoa(edge.Child), strconv.FormatFloat(edge.Lambda, 'g', -1, 64), strconv.Itoa(edge.ChildSize)}
	}, func(encoder *json.Encoder, i int) error { return encoder.Encode(result.CondensedTree[i]) })
}

// Write streams a packed branch approximation graph without materializing an
// adjacency list or an unpacked copy.
func (graph BranchGraph) Write(writer io.Writer, format ExportFormat) error {
	return writeRecords(writer, format, []string{"from", "to", "centrality", "reachability"}, graph.Len(), func(i int) []string {
		from, to, centrality, reachability := graph.Edge(i)
		return []string{strconv.Itoa(from), strconv.Itoa(to), strconv.FormatFloat(centrality, 'g', -1, 64), strconv.FormatFloat(reachability, 'g', -1, 64)}
	}, func(encoder *json.Encoder, i int) error {
		from, to, centrality, reachability := graph.Edge(i)
		return encoder.Encode(struct {
			From         int     `json:"from"`
			To           int     `json:"to"`
			Centrality   float64 `json:"centrality"`
			Reachability float64 `json:"reachability"`
		}{from, to, centrality, reachability})
	})
}

func writeRecords(writer io.Writer, format ExportFormat, header []string, count int, csvRecord func(int) []string, jsonRecord func(*json.Encoder, int) error) error {
	if writer == nil {
		return fmt.Errorf("hdbscan: nil export writer")
	}
	switch format {
	case ExportCSV:
		encoder := csv.NewWriter(writer)
		if err := encoder.Write(header); err != nil {
			return err
		}
		for i := 0; i < count; i++ {
			if err := encoder.Write(csvRecord(i)); err != nil {
				return err
			}
		}
		encoder.Flush()
		return encoder.Error()
	case ExportJSON:
		encoder := json.NewEncoder(writer)
		for i := 0; i < count; i++ {
			if err := jsonRecord(encoder, i); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("hdbscan: unsupported export format %q", format)
	}
}
