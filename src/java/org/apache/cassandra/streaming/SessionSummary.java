/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.cassandra.streaming;

import java.io.IOException;
import java.io.Serializable;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.cassandra.db.TypeSizes;
import org.apache.cassandra.io.util.DataInputPlus;
import org.apache.cassandra.io.util.DataOutputPlus;
import org.apache.cassandra.repair.messages.RepairVerbs;
import org.apache.cassandra.serializers.InetAddressSerializer;
import org.apache.cassandra.utils.ByteBufferUtil;
import org.apache.cassandra.utils.Serializer;
import org.apache.cassandra.utils.versioning.VersionDependent;
import org.apache.cassandra.utils.versioning.Versioned;

public class SessionSummary implements Serializable
{
    public static final Versioned<RepairVerbs.RepairVersion, Serializer<SessionSummary>> serializers = RepairVerbs.RepairVersion.versioned(SessionSummarySerializer::new);
    public final InetAddress coordinator;
    public final InetAddress peer;
    /** Immutable collection of receiving summaries */
    public final Collection<StreamSummary> receivingSummaries;
    /** Immutable collection of sending summaries*/
    public final Collection<StreamSummary> sendingSummaries;

    public SessionSummary(InetAddress coordinator, InetAddress peer,
                          Collection<StreamSummary> receivingSummaries,
                          Collection<StreamSummary> sendingSummaries)
    {
        assert coordinator != null;
        assert peer != null;
        assert receivingSummaries != null;
        assert sendingSummaries != null;

        this.coordinator = coordinator;
        this.peer = peer;
        this.receivingSummaries = receivingSummaries;
        this.sendingSummaries = sendingSummaries;
    }

    public boolean equals(Object o)
    {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        SessionSummary summary = (SessionSummary) o;

        if (!coordinator.equals(summary.coordinator)) return false;
        if (!peer.equals(summary.peer)) return false;
        if (!receivingSummaries.equals(summary.receivingSummaries)) return false;
        return sendingSummaries.equals(summary.sendingSummaries);
    }

    public int hashCode()
    {
        int result = coordinator.hashCode();
        result = 31 * result + peer.hashCode();
        result = 31 * result + receivingSummaries.hashCode();
        result = 31 * result + sendingSummaries.hashCode();
        return result;
    }

    public static class SessionSummarySerializer extends VersionDependent<RepairVerbs.RepairVersion> implements Serializer<SessionSummary>
    {
        protected SessionSummarySerializer(RepairVerbs.RepairVersion version)
        {
            super(version);
        }

        public void serialize(SessionSummary summary, DataOutputPlus out) throws IOException
        {
            ByteBufferUtil.writeWithLength(InetAddressSerializer.instance.serialize(summary.coordinator), out);
            ByteBufferUtil.writeWithLength(InetAddressSerializer.instance.serialize(summary.peer), out);

            out.writeInt(summary.receivingSummaries.size());
            for (StreamSummary streamSummary: summary.receivingSummaries)
            {
                StreamSummary.serializers.get(version.streamVersion).serialize(streamSummary, out);
            }

            out.writeInt(summary.sendingSummaries.size());
            for (StreamSummary streamSummary: summary.sendingSummaries)
            {
                StreamSummary.serializers.get(version.streamVersion).serialize(streamSummary, out);
            }
        }

        public SessionSummary deserialize(DataInputPlus in) throws IOException
        {
            InetAddress coordinator = InetAddressSerializer.instance.deserialize(ByteBufferUtil.readWithLength(in));
            InetAddress peer = InetAddressSerializer.instance.deserialize(ByteBufferUtil.readWithLength(in));

            int numRcvd = in.readInt();
            List<StreamSummary> receivingSummaries = new ArrayList<>(numRcvd);
            for (int i=0; i<numRcvd; i++)
            {
                receivingSummaries.add(StreamSummary.serializers.get(version.streamVersion).deserialize(in));
            }

            int numSent = in.readInt();
            List<StreamSummary> sendingSummaries = new ArrayList<>(numRcvd);
            for (int i=0; i<numSent; i++)
            {
                sendingSummaries.add(StreamSummary.serializers.get(version.streamVersion).deserialize(in));
            }

            return new SessionSummary(coordinator, peer, receivingSummaries, sendingSummaries);
        }

        public long serializedSize(SessionSummary summary)
        {
            long size = 0;
            size += ByteBufferUtil.serializedSizeWithLength(InetAddressSerializer.instance.serialize(summary.coordinator));
            size += ByteBufferUtil.serializedSizeWithLength(InetAddressSerializer.instance.serialize(summary.peer));

            size += TypeSizes.sizeof(summary.receivingSummaries.size());
            for (StreamSummary streamSummary: summary.receivingSummaries)
            {
                size += StreamSummary.serializers.get(version.streamVersion).serializedSize(streamSummary);
            }
            size += TypeSizes.sizeof(summary.sendingSummaries.size());
            for (StreamSummary streamSummary: summary.sendingSummaries)
            {
                size += StreamSummary.serializers.get(version.streamVersion).serializedSize(streamSummary);
            }
            return size;
        }
    }
}
