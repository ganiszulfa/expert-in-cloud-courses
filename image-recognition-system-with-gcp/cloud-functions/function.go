package ImageLabeling

import (
        "context"
        "fmt"
        "log"
        "os"
        "strings"
        "encoding/json"

        "cloud.google.com/go/pubsub"
        "github.com/GoogleCloudPlatform/functions-framework-go/functions"
        "github.com/cloudevents/sdk-go/v2/event"
        "cloud.google.com/go/vision/apiv1"
)

func init() {
        functions.CloudEvent("imageLabeling", imageLabeling)
}

func imageLabeling(ctx context.Context, e event.Event) error {

        newImageEvent, err := receiveEventFromPubSub(e)
        if err != nil {
                log.Printf("receiveEventFromPubSub error: %v", err)
                return err
        }

        labels := []string{}
        path := fmt.Sprintf("gs://%s/%s", os.Getenv("BUCKET_NAME"), newImageEvent.Path) 
        if labels, err = labelImageWithVisionAi(ctx, path); err != nil {
                log.Printf("labelImageWithVisionAi error: %v", err)
                return err
        }
        log.Printf("labels : %v", labels)
        
        if err = publishResultToPubSub(ctx, newImageEvent.Id, strings.Join(labels, ",")); err != nil {
                log.Printf("publishResultToPubSub error: %v", err)
                return err
        }

        return nil
}

func labelImageWithVisionAi(ctx context.Context, uri string) ([]string, error) {

        client, err := vision.NewImageAnnotatorClient(ctx)
        if err != nil {
                return nil, err
        }
        defer client.Close()

        image := vision.NewImageFromURI(uri)

        annotations, err := client.DetectLabels(ctx, image, nil, 10)
        if err != nil {
                return nil, err
        }
        var labels []string
        for _, annotation := range annotations {
                labels = append(labels, annotation.Description)
        }
        return labels, nil
}


func receiveEventFromPubSub(e event.Event) (NewImage, error) {
        var msg MessagePublishedData
        if err := e.DataAs(&msg); err != nil {
                return NewImage{}, fmt.Errorf("event.DataAs: %v", err)
        }

        data := NewImage{}

        dataStr := string(msg.Message.Data)

        if err := json.Unmarshal([]byte(dataStr), &data); err != nil {
                return NewImage{}, fmt.Errorf("unmarshal: %v", err)
        }

        return data, nil

}

func publishResultToPubSub(ctx context.Context, image_id int, labels string) error {
        client, err := pubsub.NewClient(ctx, os.Getenv("PROJECT_ID"))
        if err != nil {
                return fmt.Errorf("pubsub: NewClient: %w", err)
        }
        defer client.Close()

        record := map[string]interface{}{"image_id": image_id, "labels": labels}
        t := client.Topic(os.Getenv("TOPIC_ID"))
        result := t.Publish(ctx, &pubsub.Message{
                Data: []byte(fmt.Sprintf("%v", record)),
        })

        id, err := result.Get(ctx)
        if err != nil {
                return fmt.Errorf("pubsub: result.Get: %w", err)
        }
        log.Printf("Published a message; msg ID: %v\n", id)
        return nil
}

type MessagePublishedData struct {
        Message PubSubMessage
}

type NewImage struct {
        Id int
        Path string
}

type PubSubMessage struct {
        Data []byte `json:"data"`
}
